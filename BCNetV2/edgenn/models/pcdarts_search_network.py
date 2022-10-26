import torch
import torch.nn as nn
import torch.nn.functional as F
from .operations import *
from .genotypes import PRIMITIVES
from .genotypes import Genotype

from .builder import BackboneReg

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    
    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
 
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x

class MixedOp(nn.Module):

  def __init__(self, C, stride, k):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    self.mp = nn.MaxPool2d(2, 2)
    self.k = k
    for primitive in PRIMITIVES:
      op = OPS[primitive](C//self.k, stride, False)
      if 'pool' in primitive:
        op = nn.Sequential(op, nn.BatchNorm2d(C//self.k, affine=False))
      self._ops.append(op)

  def forward(self, x, weights):
    dim_2 = x.shape[1]
    xtemp = x[:, :dim_2//self.k, :, :]
    xtemp2 = x[:, dim_2//self.k:, :, :]
    temp1 = sum(w * op(xtemp) for w, op in zip(weights, self._ops))

    #reduction cell needs pooling before concat
    if temp1.shape[2] == x.shape[2]:
      ans = torch.cat([temp1, xtemp2],dim=1)
    else:            
      ans = torch.cat([temp1, self.mp(xtemp2)], dim=1)
    ans = channel_shuffle(ans, self.k)
    return ans

class Cell(nn.Module):

  def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, k):
    super(Cell, self).__init__()
    self.reduction = reduction

    if reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
    self._steps = steps
    self._multiplier = multiplier
    self._k = k

    self._ops = nn.ModuleList()
    self._bns = nn.ModuleList()
    for i in range(self._steps):
      for j in range(2+i):
        stride = 2 if reduction and j < 2 else 1
        op = MixedOp(C, stride, k)
        self._ops.append(op)

  def forward(self, s0, s1, weights, weights2):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]
    offset = 0
    for i in range(self._steps):
      s = sum(weights2[offset+j] * self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
      offset += len(states)
      states.append(s)

    return torch.cat(states[-self._multiplier:], dim=1)


@BackboneReg.register_module('pcdartssearchnetwork')
class PCDARTSSearchNetwork(nn.Module):

  def __init__(self, C, num_classes, layers, k, stem_type='cifar', steps=4, multiplier=4, stem_multiplier=3):
    super(PCDARTSSearchNetwork, self).__init__()
    self._C = C
    self._num_classes = num_classes
    self._layers = layers
    self._k = k
    self._steps = steps
    self._multiplier = multiplier
    assert stem_type in ('cifar', 'cifar10', 'imagenet')
    self._stem_type = stem_type

    C_curr = stem_multiplier*C
    if stem_type in ('cifar', 'cifar10'):
      self.stem = nn.Sequential(
        nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
        nn.BatchNorm2d(C_curr)
      )
      reduction_prev = False
    elif stem_type in ('imagenet',):
      self.stem0 = nn.Sequential(
        nn.Conv2d(3, C_curr // 2, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(C_curr // 2), 
        nn.ReLU(inplace=True),
        nn.Conv2d(C_curr // 2, C_curr, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(C_curr),
      )   
 
      self.stem1 = nn.Sequential(
        nn.ReLU(inplace=True),
        nn.Conv2d(C_curr, C_curr, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(C_curr),
      )
      reduction_prev = True
 
    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
    self.cells = nn.ModuleList()
    for i in range(layers):
      if i in [layers//3, 2*layers//3]:
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, k)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, multiplier*C_curr

    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)

    n = sum(1 for i in range(self._steps) for n in range(2+i))
    num_ops = len(PRIMITIVES)

    self.register_buffer('alpha_normal', (1e-3 * torch.randn(n, num_ops)).clone().detach().cuda().requires_grad_(True))
    self.register_buffer('alpha_reduce', (1e-3 * torch.randn(n, num_ops)).clone().detach().cuda().requires_grad_(True))
    self.register_buffer('beta_normal', (1e-3 * torch.randn(n)).clone().detach().cuda().requires_grad_(True))
    self.register_buffer('beta_reduce', (1e-3 * torch.randn(n)).clone().detach().cuda().requires_grad_(True))

  def forward(self, input):
    if self._stem_type in ('cifar', 'cifar10'):
      s0 = s1 = self.stem(input)
    elif self._stem_type in ('imagenet',):
      s0 = self.stem0(input)
      s1 = self.stem1(s0)

    for cell in self.cells:
      if cell.reduction:
        alpha = self.alpha_reduce
        beta = self.beta_reduce
      else:
        alpha = self.alpha_normal
        beta = self.beta_normal
      weights = F.softmax(alpha, dim=-1)
      n = 3            
      start = 2        
      weights2 = F.softmax(beta[0:2], dim=-1)
      for _ in range(self._steps-1):
        end = start + n   
        tw2 = F.softmax(beta[start:end], dim=-1)
        start = end    
        n += 1         
        weights2 = torch.cat([weights2, tw2],dim=0)
      s0, s1 = s1, cell(s0, s1, weights, weights2)
    out = self.global_pooling(s1)         
    logits = self.classifier(out.view(out.size(0), -1))
    return logits

  def arch_parameters(self):
    return [self.alpha_normal, self.alpha_reduce, self.beta_normal, self.beta_reduce]

  def genotype(self):

    def _parse(weights, weights2):
      gene = []
      n = 2
      start = 0
      for i in range(self._steps):
        end = start + n
        W = weights[start:end].copy()
        W2 = weights2[start:end].copy()
        for j in range(n):
          W[j,:]=W[j,:]*W2[j]
        edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
        for j in edges:
          k_best = None
          for k in range(len(W[j])):
            if k != PRIMITIVES.index('none'):
              if k_best is None or W[j][k] > W[j][k_best]:
                k_best = k
          gene.append((PRIMITIVES[k_best], j))
        start = end
        n += 1
      return gene

    n = 3        
    start = 2    
    weightsn2 = F.softmax(self.beta_normal[0:2], dim=-1)
    weightsr2 = F.softmax(self.beta_reduce[0:2], dim=-1)
    for i in range(self._steps-1):
      end = start + n
      tn2 = F.softmax(self.beta_normal[start:end], dim=-1)
      tw2 = F.softmax(self.beta_reduce[start:end], dim=-1)
      start = end
      n += 1     
      weightsn2 = torch.cat([weightsn2,tn2],dim=0)
      weightsr2 = torch.cat([weightsr2,tw2],dim=0)
    gene_normal = _parse(F.softmax(self.alpha_normal, dim=-1).data.cpu().numpy(), weightsn2.data.cpu().numpy())
    gene_reduce = _parse(F.softmax(self.alpha_reduce, dim=-1).data.cpu().numpy(), weightsr2.data.cpu().numpy())

    concat = range(2+self._steps-self._multiplier, self._steps+2)
    genotype = Genotype(
      normal=gene_normal, normal_concat=concat,
      reduce=gene_reduce, reduce_concat=concat
    )
    return genotype

