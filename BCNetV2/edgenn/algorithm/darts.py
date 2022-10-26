import copy
import torch
import numpy as np
import torch
import torch.nn as nn

from .base import BaseAlgorithm
from ..utils import AlgorithmReg

def _concat(xs):
  return torch.cat([x.view(-1) for x in xs])


@AlgorithmReg.register_module('darts')
class DARTSAlgorithm(BaseAlgorithm):

  def __init__(self, unrolled, optimizer):
    self.unrolled = unrolled
    self.arch_optimizer = None
    self.arch_optimizer_cfg = optimizer

  def step(self, model, optimizer, input_train, target_train, input_valid, target_valid):
    if self.arch_optimizer is None:
        #self.arch_optimizer = build_optimizer(model.module.arch_parameters(), self.arch_optimizer_cfg)
        assert self.arch_optimizer_cfg.type == 'Adam'
        self.arch_optimizer = torch.optim.Adam(model.module.arch_parameters(),
            lr=self.arch_optimizer_cfg.lr, betas=self.arch_optimizer_cfg.betas, weight_decay=self.arch_optimizer_cfg.weight_decay)
    self.arch_optimizer.zero_grad()
    eta = optimizer.param_groups[0]['lr']
    if self.unrolled:
        assert optimizer is not None, 'Optimizer must be set in unrolled mode.'
        self._backward_step_unrolled(model, optimizer, input_train, target_train, input_valid, target_valid, eta)
    else:
        self._backward_step(model, input_valid, target_valid)
    self.arch_optimizer.step()

  def _backward_step(self, model, input_valid, target_valid):
    output, loss = model(input_valid, target_valid)
    loss.backward()

  def _backward_step_unrolled(self, model, optimizer, input_train, target_train, input_valid, target_valid, eta):
    unrolled_model = copy.deepcopy(model)
    unrolled_optimizer = copy.deepcopy(optimizer)
    unrolled_optimizer.param_groups.clear()
    unrolled_optimizer.state.clear()
    unrolled_optimizer.add_param_group({'params': unrolled_model.parameters()})
    unrolled_optimizer.load_state_dict(optimizer.state_dict())

    for param in unrolled_model.module.arch_parameters():
        param.requires_grad = False
    output, loss = unrolled_model(input_train, target_train)
    unrolled_optimizer.zero_grad()
    loss.backward()
    unrolled_optimizer.step()

    for param in unrolled_model.module.arch_parameters():
        param.requires_grad = True
    output, loss = unrolled_model(input_valid, target_valid)
    unrolled_optimizer.zero_grad()
    loss.backward()
    dalpha = [v.grad for v in unrolled_model.module.arch_parameters()]
    vector = [v.grad for v in unrolled_model.parameters()]

    implicit_grads = self._hessian_vector_product(model, vector, input_train, target_train)
    for g, ig in zip(dalpha, implicit_grads):
      g.data.sub_(eta, ig)

    for v, g in zip(model.module.arch_parameters(), dalpha):
        v.grad = g

  def _hessian_vector_product(self, model, vector, input, target, r=1e-2):
    R = r / _concat(vector).norm()

    for p, v in zip(model.parameters(), vector):
      p.data.add_(R, v)
    output, loss = model(input, target)
    grads_p = torch.autograd.grad(loss, model.module.arch_parameters())

    for p, v in zip(model.parameters(), vector):
      p.data.sub_(2*R, v)
    output, loss = model(input, target)
    grads_n = torch.autograd.grad(loss, model.module.arch_parameters())

    for p, v in zip(model.parameters(), vector):
      p.data.add_(R, v)

    return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]
