# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 10:08:43 2022

@author: mthibode
"""

import pytest
import math
import numpy as np
import quimb as qu
import quimb.tensor as qtn
from ttn import TTN, energy, local_ham, sweep_tree, evaluate_move, propose_move, anneal_timestep

L = 9
phys_dim = 2
min_bond = 8
max_bond = 16


def test_random_init():
    
    x = TTN.random_TTN(L, phys_dim, min_bond, max_bond)
    physical_inds = x.outer_inds()
    
    for tensor in x.tensors:
        for i in tensor.inds:
            s = tensor.ind_size(i)
            if i in physical_inds:
                assert s == phys_dim
            else:
                assert s <= max_bond
                
def test_copy():
    x = TTN.random_TTN(L, phys_dim, min_bond, max_bond)
    y = x.copy()
    
    k,v = zip(*x._parents.items())
    x._parents[k[0]] = 'nonsense'
    
    assert y._parents[k[0]] != 'nonsense'
    

def test_get_tree_tag():
    x = TTN.random_TTN(L, phys_dim, min_bond, max_bond)
    tensor = x.tensors[0]
    tt = x.get_tree_tag(tensor)
    
    assert ('tree' in tt)
    
def test_tree_name():
    x = TTN.random_TTN(L, phys_dim, min_bond, max_bond)
    tensor = x.tensors[0]
    tt = x.get_tree_tag(tensor)
    
    assert x.tree_name(tensor) == tt
    assert x.tree_name(tt) == tt
    
def test_get_tree_num():
    x = TTN.random_TTN(L, phys_dim, min_bond, max_bond)
    tt = 'tree0'
    tensor = x[tt]
    
    assert 0 == x.get_tree_num(tensor)
    assert 0 == x.get_tree_num(tt)


def test_get_adjacency_matrix():
    x = TTN.random_TTN(4, 9, 18, 18)
      
    assert np.all(x.get_adjacency_matrix() == np.array(
        [[ 0,  0,  0,  0,  9,  0,  0],
           [ 0,  0,  0,  0,  9,  0,  0],
           [ 0,  0,  0,  0,  0,  9,  0],
           [ 0,  0,  0,  0,  0,  9,  0],
           [ 9,  9,  0,  0,  0,  0, 18],
           [ 0,  0,  9,  9,  0,  0, 18],
           [ 0,  0,  0,  0, 18, 18,  0]]))
    

def test_get_parent():

    # tree must be binary
    x = TTN.random_TTN(L, phys_dim, min_bond, max_bond)
    parent = x.get_parent(x['tree0'])
    assert parent == f'tree{L}'
    
def test_set_parent():
    
    # tree must be binary
    x = TTN.random_TTN(L, phys_dim, min_bond, max_bond)
    x.set_parent('tree0', f'tree{L-1}')
    
    assert x.get_parent('tree0') == f'tree{L-1}'
    
def test_get_all_parents():
    
    # L must be a power of two, and tree must be binary
    height = 4
    L = 2 ** height
    x = TTN.random_TTN(L, phys_dim, min_bond, max_bond)
    all_parents = x.get_all_parents('tree0')
    nums = [L // (2 **k) for k in range(height)]
    expected_nums = [sum(nums[:k]) for k in range(1, len(nums) + 1)]
    expected = [f'tree{k}' for k in expected_nums]
    
    assert list(sorted(all_parents)) == list(sorted(expected))
    

def test_get_children():
    
    # tree must be binary
    x = TTN.random_TTN(L, phys_dim, min_bond, max_bond)
    
    assert list(x.get_children(f'tree{L}')) == ['tree0', 'tree1']
    

def test_get_descendents():
    
    # tree must be binary
    x = TTN.random_TTN(L, phys_dim, min_bond, max_bond)
    gplabel = x.get_parent(x.get_parent('tree0'))
    grandchildren = list(x.get_descendents(gplabel))
    
    assert grandchildren == [f'tree{k}' for k in range(4)]
    
def test_move_one_link():
    
    x = TTN.random_TTN(L, phys_dim, min_bond, max_bond)
    gp = x.get_parent(x.get_parent('tree0'))

    x.move_one_link(x['tree0'], x.get_parent('tree0'), 
                    gp)
    
    assert x.get_parent('tree0') == gp
    
def test_fuse_subtrees():
    
    # tree must be ternary
    height = 3
    L = 3 ** height
    x = TTN.random_TTN(L, phys_dim, min_bond, max_bond, 4, 4)
    x.fuse_subtrees(x.get_parent('tree0'), x.get_parent('tree3'))
    
    assert x.get_parent(x.get_parent(x.get_parent('tree0'))) == x.get_parent(x.get_parent('tree6'))
    
    
    
def test_canonize_between():
    
    x = TTN.random_TTN(L, phys_dim, min_bond, max_bond)
    x.canonize_between('tree0', f'tree{L}')
    
    t = x['tree0']
    assert round(t @ t.H) == phys_dim
    
    
def test_canonize_subtree():
    
    x = TTN.random_TTN(L, phys_dim, min_bond, max_bond)
    gp = x.get_parent(x.get_parent('tree0'))
    x.canonize_subtree(gp)
    t = x[gp]
    assert round(t @ t.H) == max(t.shape)
    

def test_normalize():
    x = TTN.random_TTN(L, phys_dim, min_bond, max_bond)
    x.normalize()
    assert round(x @ x.H) == 1

def test_get_leaves():
    x = TTN.random_TTN(L, phys_dim, min_bond, max_bond)
    leaves = list(sorted(x.get_leaves()))
    # expected = [f'tree{k}' for k in range(L)]
    for leaf in leaves:
        assert list(x.get_children(leaf)) == []
    
    
def test_get_top_parent():
    x = TTN.random_TTN(L, phys_dim, min_bond, max_bond)
    tp = x.get_top_parent()
    
    assert x.get_parent(tp) == None
    
def test_get_path():
    
    x = TTN.random_TTN(L, phys_dim, min_bond, max_bond)
    last_leaf = list(x.get_leaves())[-1]
    path = list(x.get_path('tree0', last_leaf))
    expected = ['tree0'] + list(x.get_all_parents('tree0')) + list(reversed(
        x.get_all_parents(last_leaf)))[1:] + [last_leaf]
    assert path == expected

                
def test_transport_subtree():
    
    x = TTN.random_TTN(L, phys_dim, min_bond, max_bond)
    
    parent = x._parents['tree0']
    grandparent = x._parents[parent]
    cousins = x.get_children(grandparent)
    
    target = next((x for x in cousins if x != parent))
    predegree = len(list(x.get_children(target)))
    
    x.transport_subtree('tree0', target)
    postdegree = len(list(x.get_children(target)))
    
    assert predegree + 1 == postdegree
    assert ('tree0' in x.get_children(target))
    
def test_local_ham():

    A = qtn.MPO_rand_herm(L, bond_dim=10, tags=['_HAM'])
    x = TTN.random_TTN(L, phys_dim, min_bond, max_bond)
    test_site = 'tree0'
    
    h, hm, ki, bi, xi = local_ham(A, x, test_site)
    
    assert list(h.shape) == [phys_dim] * 4
    
def test_sweep_tree():
    
    # L shouldn't be too large for performance reasons
    Lmin = min(L, 5)
    
    A = qtn.MPO_rand_herm(Lmin, bond_dim=10, tags=['_HAM'])
    x = TTN.random_TTN(Lmin, phys_dim, min_bond, max_bond)
    
    e0 = energy(x, A) ^ ...
    
    energies = sweep_tree(x, A, L, max_bond)
    
    assert energies[-1] <= e0
    
def test_propose_move():
    
    x = TTN.random_TTN(L, phys_dim, min_bond, max_bond)
    xp, moved_node, old_parent, new_parent = propose_move(x)
    
    assert x.get_parent(moved_node) == old_parent
    assert xp.get_parent(moved_node) == new_parent
    

    
    
    

    
    
    


