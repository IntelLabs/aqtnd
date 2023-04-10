#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 15:37:15 2023

@author: matthewthibodeau
"""
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 13:28:31 2022

@author: mthibode
"""

import numpy as np
import matplotlib.pyplot as plt
import quimb as qu
import quimb.tensor as qtn
import cotengra as ctg

from warnings import simplefilter
simplefilter("ignore", category=FutureWarning)

from math import cos, sin, pi

opt = ctg.ReusableHyperOptimizer(
    progbar=True,
    reconf_opts={},
    max_repeats=16,
    # directory=  # set this for persistent cache
)
def norm_fn(mera):
    # there are a few methods to do the projection
    # exp works well for optimization
    return mera.unitize(method='exp')

def local_expectation(mera, terms, where, optimize='auto-hq'):
    """Compute the energy for a single local term.
    """
    # get the lightcone for `where`
    tags = [mera.site_tag(coo) for coo in where]
    mera_ij = mera.select(tags, 'any')

    # apply the local gate
    G = terms[where]
    mera_ij_G = mera_ij.gate(terms[where], where)

    # compute the overlap - this is where the real computation happens
    mera_ij_ex = (mera_ij_G & mera_ij.H)
    return mera_ij_ex.contract(all, optimize=optimize)


def loss_fn(mera, ref, optimize='auto-hq'):
    """Compute the total energy as a sum of all terms.
    """
    return - ((ref & mera).contract(all, optimize = optimize))


def loss_energy(mera, terms, **kwargs):
    """Compute the total energy as a sum of all terms.
    """
    return sum(
        local_expectation(mera, terms, where, **kwargs)
        for where in terms
    )

def mera_lowest_connector_uni(mera, site1, site2):
    # mera.draw_tree_span()
    connector = mera.select([site1, site2])
    tags = connector.tags
    minlayer = min([int(x[-1]) for x in tags if '_LAYER' in x])
    # could be off by 1
    
    minunilist = connector.select(['_LAYER' + str(minlayer), '_UNI']).tensors
    if len(minunilist) == 0:
        minunilist = connector.select(['_LAYER' + str(minlayer+1), '_UNI']).tensors
    minuni = minunilist[0]
    

opt = ctg.ReusableHyperOptimizer(
    progbar=True,
    reconf_opts={},
    max_repeats=16,
    # directory=  # set this for persistent cache
)



# total length (currently must be power of 2)
L = 2**4

# max bond dimension
mD = 6
xD = 5

# use single precision for quick GPU optimization
dtype = 'float32'

# start with a bond dimension of 2
mera = qtn.MERA.rand(L, max_bond=2, dtype=dtype)

# this is the function that projects all tensors
# with ``left_inds`` into unitary / isometric form
mera.unitize_()

disorder_strength = 0.5
j0 = 1
coupling_vals = np.random.normal(j0, disorder_strength, L)

H2 = qu.ham_heis(2).real.astype(dtype)
terms = {((i - 1) % L, i): coupling_vals[i] * H2 for i in range(L)}

builder = qtn.SpinHam1D(S=1/2, cyclic=True)
for i in range(1, L):
    builder[i-1, i] += coupling_vals[i], 'X', 'X'
    builder[i-1, i] += coupling_vals[i], 'Y', 'Y'
    builder[i-1, i] += coupling_vals[i], 'Z', 'Z'
    
builder[L, L+1] += coupling_vals[0], 'X', 'X'
builder[L, L+1] += coupling_vals[0], 'Y', 'Y'
builder[L, L+1] += coupling_vals[0], 'Z', 'Z'

Htotal = builder.build_mpo(L)

x = qtn.MPS_rand_state(L, xD, cyclic=True, dtype=dtype)
x.compress()


# do the renormalization to get singlets
def singletize_H(Htotal):
    ...
    # returns a list of singlet indices [..., (left, right) ,...]

singlets = singletize_H(Htotal)

# now expand the bond dimensions of the mera
for left, right in singlets:
    minuni = mera_lowest_connector_uni(mera, left, right)
    unisize = minuni.shape[0]
    inds = minuni.inds
    mera.expand_bond_dimension(unisize + 1, rand_strength = 1.0, 
                               inds_to_expand=minuni.inds, inplace=True)
    # for k in range(len(inds)):
    #     # len should be 4
    #     mera.expand_bond_dimension(minuni)


