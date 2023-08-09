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
    x =  mera.unitize(method='exp')
    return x/np.sqrt(x @ x.H)

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

mera = qtn.MERA.rand(L, max_bond=mD, dtype=dtype)

# this is the function that projects all tensors
# with ``left_inds`` into unitary / isometric form
mera.unitize_()

fix = {
    f'k{i}': (sin(2 * pi * i / L), cos(2 * pi * i / L))
    for i in range(L)
}

# reduce the 'spring constant' k as well
draw_opts = dict(fix=fix, k=0.01)
# mera.draw(color=['I0', 'I40'], **draw_opts)

H2 = qu.ham_heis(2).real.astype(dtype)
terms = {(i, (i + 1) % L): H2 for i in range(L)}

Htotal = qtn.MPO_ham_heis(L, cyclic = True)

x = qtn.MPS_rand_state(L, xD, cyclic=False, dtype=dtype)
x.compress()
xH = x.H

tnopt = qtn.TNOptimizer(
    mera,
    loss_fn=loss_fn,
    norm_fn=norm_fn,
    loss_constants={'ref': xH},
    loss_kwargs={'optimize': opt},
    autodiff_backend='torch', jit_fn=True,
)
tnopt.optimizer = 'l-bfgs-b'  # the default
# mo = tnopt.optimize(1)

eopt = qtn.TNOptimizer(
    mera,
    loss_fn=loss_energy,
    norm_fn=norm_fn,
    loss_constants={'terms': terms},
    loss_kwargs={'optimize': opt},
    autodiff_backend='torch', jit_fn=True,
)
eopt.optimizer = 'l-bfgs-b'  # the default
emo = eopt.optimize(1)


# loss_fn(mera, xH, opt)

# energy_tol = 1e-6
# dmrg = qtn.DMRG2(Htotal, bond_dims = [5, 6, 7, 8, 9, 10])
# succ = dmrg.solve(energy_tol, max_sweeps = 10, sweep_sequence = 'RL', verbosity = 1)
# edmrg = dmrg.energy









