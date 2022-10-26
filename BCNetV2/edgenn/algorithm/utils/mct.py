import math
import random
import torch
import torch.distributed as dist


def _UCT(reward, visits, parent_visits, C1, C2=0, G=0):
    if visits != 0:
        r0 = reward / visits
    else:
        r0 = 0
    if parent_visits != 0:
        r1 = math.sqrt(math.log(parent_visits) / (visits + 0.1))
    else:
        r1 = 0
    uct = r0 + C1 * r1 + C2 * G
    return uct


class MCTree:
    def __init__(self, choices, gamma=0.9, C1=1., C2=0., tau=0.0025):
        self.root = MCNode(None, choices[0])
        self.choices = choices
        self.total_depth = len(choices)
        self.gamma = gamma
        self.C1 = C1
        self.C2 = C2
        self.tau = tau
        self.G = []
        for op_num in choices:
            self.G.append([0] * op_num)

    @property
    def size(self):
        '''get size (number of leaf nodes) of the tree'''
        # depth first search
        num_leaves = 0
        def traverse(node, depth):
            if node is None:
                return
            if depth == self.total_depth and node.visits > 0:
                nonlocal num_leaves
                num_leaves += 1
                return
            for child in node.children:
                traverse(child, depth+1)

        traverse(self.root, 0)
        return num_leaves

    def update(self, subnet, reward):
        node = self.root
        node.update(reward)
        for depth, idx in enumerate(subnet):
            if node[idx] is None:
                node.add_child(idx, MCNode(node, self.choices[depth+1] if depth != self.total_depth -1 else 0))
            node = node[idx]
            node.update(reward)

            # update G
            self.G[depth][idx] = self.G[depth][idx] * self.gamma + (1 - self.gamma) * reward

    def sample(self, ignore_none=False):
        subnet = []
        node = self.root
        for depth in range(self.total_depth):
            idx = node.sample_child(self.C1, self.C2, self.G[depth], self.tau)
            subnet.append(idx)
            if node[idx] is None:
                node.add_child(idx, MCNode(node, self.choices[depth+1] if depth != self.total_depth -1 else 0))
            node = node[idx]
        return subnet

    def remove_leaf_node(self, subnet):
        node = self.root
        for depth, idx in enumerate(subnet[:-1]):
            node = node[idx]
            if node is None:
                return
        
        node[subnet[-1]] = None

        for depth in range(self.total_depth-2, -1, -1):
            if node.children.count(None) == len(node.children):
                node = node.parent
                node[subnet[depth]] = None
            else:
                break
            
    def _broadcast_subnet(self, subnet):
        subnet = torch.tensor(subnet, dtype=torch.int32, device='cuda')
        dist.broadcast(subnet, src=0)
        return subnet.tolist()

    def _hierarchical_node_update(self, start_depth, subnet, eval_fn, flops_limit, tilde_L):
        subnet = subnet.copy()
        if len(subnet) != self.total_depth:
            subnet += [None] * (self.total_depth - len(subnet))
        node = self.root
        for depth in range(self.total_depth):
            if depth < start_depth:
                node = node[subnet[depth]]
            else:
                idx = node.sample_child_uniformly(ignore_none=True)
                #if node[idx] is None:
                #    node.add_child(idx, MCNode(node, self.choices[depth+1] if depth != self.total_depth -1 else 0))
                subnet[depth] = idx
                node = node[idx]
        subnet = self._broadcast_subnet(subnet)
        # evaluate
        flops = eval_fn(subnet, flops=True)
        flops_min, flops_max = flops_limit
        if (flops_max != -1 and flops > flops_max) or flops < flops_min: 
            self.remove_leaf_node(subnet)
            return
        score, flops = eval_fn(subnet)
        loss = -score
        reward = tilde_L / loss
        self.update(subnet, reward)


    def hierarchical_sample(self, n_thrd, eval_fn, flops_limit, tilde_L):
        subnet = []
        node = self.root
        for depth in range(self.total_depth):
            while node.visits / self.choices[depth] < n_thrd:
                # hierarchical node update
                self._hierarchical_node_update(depth, subnet, eval_fn, flops_limit, tilde_L)
                if node.children.count(None) == len(node.children):
                    return self.hierarchical_sample(n_thrd, eval_fn, flops_limit, tilde_L)
            idx = node.sample_child(0, self.C2, self.G[depth], self.tau, ignore_none=True)
            idx = self._broadcast_subnet(idx)
            subnet.append(idx)
            node = node[idx]
        return subnet 

    def __repr__(self):
        _str = 'MCTree(choices={self.choices})'
        return _str
        

class MCNode:
    def __init__(self, parent=None, children=1):
        self.parent = parent
        self.children = [None, ] * children

        self.visits = 0
        self.reward = 0

    def add_child(self, idx, child):
        self.children[idx] = child

    def update(self, reward):
        self.reward += reward
        self.visits += 1

    def reset(self):
        self.reward = 0
        self.visits = 0
        self.children = [None, ] * len(self.children)

    def sample_child(self, C1, C2, G_l, tau=0.0025, softmax_distribution=True, ignore_none=False):
        ucts = []
        for idx, child in enumerate(self.children):
            if child is None:
                ucts.append(_UCT(0, 0, self.visits, C1, C2, G_l[idx]))
            else:
                ucts.append(_UCT(child.reward, child.visits, self.visits, C1, C2, G_l[idx]))

        if not softmax_distribution:
            if ignore_none:
                idx = ucts.index(max([uct for uct, child in zip(ucts, self.children) if child is not None]))
            else:
                idx = ucts.index(max(ucts))
        else:
            max_uct = max(ucts)
            ucts = [uct - max_uct for uct in ucts]  # avoid math overflow
            probs = [math.exp(uct / tau) for uct in ucts]
            probs_sum = sum(probs)
            probs = [x / probs_sum for x in probs]
            if ignore_none:
                probs = [0 if child is None else prob for prob, child in zip(probs, self.children)]
                probs_sum = sum(probs)
                probs = [x / probs_sum for x in probs]

            prob = random.random()
            for i in range(len(probs)):
                if prob < sum(probs[:i+1]):
                    idx = i
                    break
        return idx

    def sample_child_uniformly(self, ignore_none=False):
        if ignore_none:
            valid_indices = [idx for idx in range(len(self.children)) if self[idx] is not None]
        else:
            valid_indices = list(range(len(self.children)))
        return random.choice(valid_indices)

    def __getitem__(self, idx):
        '''
        get child
        '''
        return self.children[idx]

    def __setitem__(self, idx, value):
        self.children[idx] = value

    def __repr__(self):
        return f'MCNode(has_parent={self.parent is not None}, children={len(self.children)}, reward={self.reward}, visits={self.visits})'


