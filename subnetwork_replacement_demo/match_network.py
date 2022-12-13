# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 12:55:07 2022

@author: mthibode
"""

import numpy as np
import quimb as qu
import quimb.tensor as qtn
from functools import reduce
# import jax

# test scheme:
# target = length-L chain of degree-4 tensors with OBC (ends are degree-3), bond dim = bd
# ansatz = length-L chain, degree-4, PBC, BD = bd - k

L = 10
bd = 8
pdim = 3

target_data = ([qu.randn(shape = (bd, pdim, pdim))] + [qu.randn(shape = (bd, bd, pdim, pdim)) for _ in range(L-2)]
               + [qu.randn(shape = (bd, pdim, pdim))])
target_data = [x/np.sqrt(np.linalg.norm(x)) for x in target_data]
target_link_idxes = [qtn.rand_uuid() for _ in range(L-1)]
target_tensors = [qtn.Tensor(data = target_data[k], 
                             inds=(target_link_idxes[k-1],target_link_idxes[k], f'k{k}u', f'k{k}d'))
                  for k in range(1, L-1)]
target_tensors = ([qtn.Tensor(data = target_data[0], 
                             inds=(target_link_idxes[0], 'k0u', 'k0d'))] + target_tensors + 
                 [qtn.Tensor(data = target_data[L-1], inds=(target_link_idxes[L-2], f'k{L-1}u', f'k{L-1}d'))])
target_tn = qtn.TensorNetwork(target_tensors)#reduce(lambda x,y: x & y, target_tensors)
target_tn = target_tn / np.sqrt(target_tn.H @ target_tn)


bd_a = 8
ansatz_data = [qu.randn(shape = (bd_a, bd_a, pdim, pdim)) for _ in range(L)]
ansatz_data = [x/np.sqrt(np.linalg.norm(x)) for x in ansatz_data]
ansatz_link_idxes = [qtn.rand_uuid() for _ in range(L)]
ansatz_tensors = [qtn.Tensor(data = ansatz_data[k], 
                              inds=(ansatz_link_idxes[(k-1) % L],ansatz_link_idxes[k % L], f'k{k}u', f'k{k}d'))
                  for k in range(L)]
ansatz_tn =  qtn.TensorNetwork(ansatz_tensors)
ansatz_tn = ansatz_tn / np.sqrt(ansatz_tn.H @ ansatz_tn)
# mps = qtn.MPS_rand_state(L, bd_a, cyclic=True)



def loss_fn(network, ref):
    """Compute the total energy as a sum of all terms.
    """
    return - (ref @ network) ** 2

def norm_fn(network):
    # there are a few methods to do the projection
    # exp works well for optimization
    return network /(network.H @ network) ** 0.5

tH = target_tn.H


tnopt = qtn.TNOptimizer(
    ansatz_tn,
    loss_fn=loss_fn,
    norm_fn=norm_fn,
    loss_constants={'ref': tH},
    autodiff_backend='torch', jit_fn=True,
)

tnopt.optimizer = 'adam'  # the default
mo = tnopt.optimize(5000)
