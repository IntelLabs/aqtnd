# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 18:32:40 2022

@author: mthibode
"""

import os
import sys
import time
import numpy as np
import quimb.tensor as qtn
import torch
from torch import nn
from exact_guided_dmrg import DMRG_guided

def random_1d_ising(L, Jvar, Jmean, hvar, hmean):
    """
    Samples a random ising hamiltonian with gaussian couplings.
    Parameters
    ----------
    L : int
        number of sites in the system
    Jvar : float >= 0
        variance of the J interaction
    Jmean : float
        mean of J
    hvar : float >= 0
        variance of onsite field h
    hmean : float
        mean of j

    Returns
    -------
    quimb MPO corresponding to sampling H = sum_{j=1}^{L-1} J_j s^z_j s^z{j+1}
                                            + sum_j h_j s^x_j

    """

    jvals = np.random.normal(loc = Jmean, scale = np.sqrt(Jvar), size = L - 1)
    hvals = np.random.normal(loc = hmean, scale = np.sqrt(hvar), size = L)

    builder = qtn.SpinHam1D(S = 1./2)
    builder.build_local_ham(L)
    for j in range(L-1):
        builder[j,j+1] += jvals[j], 'Z', 'Z'
        builder[j] += hvals[j], 'X'
    builder[L-1] += hvals[L-1], 'X'

    return builder.build_mpo(L), jvals, hvals

def random_1d_heis(L, Jvar, Jmean, hvar, hmean, S):
    """
    Samples a random XXX hamiltonian with gaussian couplings.
    Parameters
    ----------
    L : int
        number of sites in the system
    Jvar : float >= 0
        variance of the J interaction
    Jmean : float
        mean of J
    hvar : float >= 0
        variance of onsite field h
    hmean : float
        mean of j

    Returns
    -------
    quimb MPO corresponding to sampling H = sum_{j=1}^{L-1} J_j s_j \cdot s_{j+1}
                                            + sum_j h_j s^z_j

    """

    jvals = np.random.normal(loc = Jmean, scale = np.sqrt(Jvar), size = L - 1)
    hvals = np.random.normal(loc = hmean, scale = np.sqrt(hvar), size = L)

    builder = qtn.SpinHam1D(S = S)
    builder.build_local_ham(L)
    for j in range(L-1):
        for x in ('X','Y','Z'):
            builder[j,j+1] += jvals[j], x, x
        builder[j] += hvals[j], 'Z'
    builder[L-1] += hvals[L-1], 'Z'

    return builder.build_mpo(L), jvals, hvals


def random_1d_heis_cos(L, Jvar, Jmean, hvar, hmean, cos_k, S):
    """
    Samples a random XXX hamiltonian with gaussian couplings modulated by cos(kx/L)
    Parameters
    ----------
    L : int
        number of sites in the system
    Jvar : float >= 0
        variance of the J interaction
    Jmean : float
        mean of J
    hvar : float >= 0
        variance of onsite field h
    hmean : float
        mean of j
    cos_k : float
        argument of cos(*x/L)

    Returns
    -------
    quimb MPO corresponding to sampling H = sum_{j=1}^{L-1} J_j S_j . S_{j+1}
                                            + sum_j h_j s^z_j

    """
    jrange = np.arange(0, L-1)
    jvals = (np.random.normal(loc = Jmean, scale = np.sqrt(Jvar), size = L - 1)
             * np.cos(cos_k * jrange/ float(L)))
    hvals = np.random.normal(loc = hmean, scale = np.sqrt(hvar), size = L)

    builder = qtn.SpinHam1D(S = S)
    builder.build_local_ham(L)
    for j in range(L-1):
        for x in ('X','Y','Z'):
            builder[j,j+1] += jvals[j], x, x
        builder[j] += hvals[j], 'Z'
    builder[L-1] += hvals[L-1], 'Z'

    return builder.build_mpo(L), jvals, hvals

def truncate_bond_to_tol(psi, left_site, loss, tol, verbose = False):
    """


    Parameters
    ----------
    psi : quimb MPS
        state to truncate
    left_site : int
        left site of bond of mps to truncate
    loss : function mps, q, r -> float
        defines the constraint loss(psi', q, r) < tol
        q and r are the modified tensors below
    tol : float
        sets the constraint level
    verbose : bool

    Returns
    -------
    quimb MPS psi', a truncated version of psi

    """

    psi.canonize(left_site)

    tl, tr = psi[left_site],  psi[left_site + 1]
    _, lix = tl.filter_bonds(tr)

    phi = psi.copy()
    pc = phi.copy()

    # default cutoff
    this_cutoff = 1e-16

    l = loss(phi, tl, tr)
    if verbose:
        print(f'cutoff: {this_cutoff:.2e}, loss: {l:.2e}')
    while l < tol and this_cutoff < 1:
        phi = pc.copy()

        if verbose:
            print(f'cutoff: {this_cutoff:.2e}, loss: {l:.2e}')

        q,r = tl.split(lix, cutoff=this_cutoff, cutoff_mode='sum2', absorb='left')

        q.transpose_like_(tl)
        r = r @ tr
        r.transpose_like_(tr)

        pc[left_site].modify(data=q.data)
        pc[left_site + 1].modify(data = r.data)

        l = loss(pc, q, r)
        this_cutoff *= 5


    return phi

def compress_mps_to_tol(psi, tol, site_loss, stop = None, verbose = 0):
    """

    Parameters
    ----------
    psi : quimb MPS
        state to be optimized
    tol : float
        overlap constraint: <psi|phi> > 1 - tol
    site_loss : function site -> function loss
        defines the site-specific constraint loss = site_loss(site)
    stop : int
        site to stop at (goes left to right)
    verbose : int
        0 == least, 1 == medium, 2 == most
    Returns
    -------
    compressed approximation phi

    """

    L = len(psi.sites)
    if stop is None:
        stop = L-1
    phi = psi.copy()
    for site in range(stop):

        psi.canonize(site)
        phi.canonize(site)


        # site_loss = lambda m, q, r: 1 - np.abs((psi[site] @ psi[site+1]).H @ (q @ r))
        phi = truncate_bond_to_tol(phi, site, site_loss(site), tol, verbose == 2)


        # phi.compress()

        tl = phi[site].copy()
        tr = phi[site+1].copy()
        for idx in tl.inds:
            tl.expand_ind(idx, psi[site].ind_size(idx))
        for idx in tr.inds:
            tr.expand_ind(idx, psi[site+1].ind_size(idx))
        phi[site].modify(data = tl.data)
        phi[site+1].modify(data = tr.data)

        if verbose > 0:
            print(f'done site {site}')


    return phi


def compress_to_overlap_tol(psi, tol, stop = None, verbose = 0):
    """

    Parameters
    ----------
    psi : quimb MPS
        state to be optimized
    tol : float
        overlap constraint: <psi|phi> > 1 - tol
    stop : int
        site to stop at (goes left to right)
    verbose : int
        0 == least, 1 == medium, 2 == most
    Returns
    -------
    compressed approximation phi that satisfies the overlap constraint

    """
    site_loss = lambda site: (lambda m, q, r:  1 - np.abs((psi[site] @ psi[site+1]).H @ (q @ r)))
    return compress_mps_to_tol(psi, tol, site_loss, stop, verbose)

def compress_to_energy_tol(psi, H, tol, stop = None, verbose = 0):
    """

    Parameters
    ----------
    psi : quimb MPS
        state to be optimized
    H : quimb MPO
        Hamiltonian to optimize against
    tol : float
        energy constraint: <phi| H |phi> < <psi|H|psi> + tol
    stop : int
        site to stop at (goes left to right)
    verbose : int
        0 == least, 1 == medium, 2 == most
    Returns
    -------
    compressed approximation phi that satisfies the overlap constraint

    """

    # psiH = psi.H
    # psi.align_(H, psiH)

    # def loss(m, *_):
    #     mH = m.H
    #     m.align_(H, mH)
    #     return ((mH & H & m) ^ ...) -( (psiH & H & psi) ^ ...)
    site_loss = lambda _ : lambda m, *_: qtn.expec_TN_1D(m.H, H, m) - qtn.expec_TN_1D(psi.H, H, psi)
    return compress_mps_to_tol(psi, tol, site_loss, stop, verbose)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(L * 2, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, L))

    def forward(self, x):
        #print(f'x: {x}')
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = torch.load('cos_model.tt')

runs = 100
for run_id in range(runs):

    Jm = -1
    Jvar = 0.1
    hm = 0
    hvar = 0.1

    cos_low = 1
    cos_high = 5

    L = int(sys.argv[1])
    site_dim = int(sys.argv[2])
    S = float(site_dim - 1)/2.

    this_id =  (os.environ['SLURM_JOB_ID']  + '_' + os.environ['SLURM_PROCID']
                + '_' + str(run_id) + f'_L{L}_D{site_dim}')
    print('starting '  + this_id)

    energy_tol = 1e-8
    svd_cutoffs = 1e-11

    #H, j, h = random_1d_heis(L, Jvar, Jm, hvar, hm, S)
    H, j, h = random_1d_heis_cos(L, Jvar, Jm, hvar, hm,
                                 np.random.randint(cos_low, cos_high), S)
    tdn = np.zeros((L,2))
    tdn[:-1, 0] = j
    tdn[:,1] = h
    torch_data = torch.Tensor(tdn).unsqueeze(0)

    dmrg = qtn.DMRG2(H, bond_dims=[20] * 10 + [30] * 10 + [40] * 10
                     + [50] * 10 + [80] * 20 + [120] * 20, cutoffs=svd_cutoffs)
    t2_i = time.perf_counter()
    # dmrg = qtn.DMRG2(H, bond_dims=[50] * 10 + [100] * 10 + [150] * 10, cutoffs=1e-8)
    succ = dmrg.solve(energy_tol, max_sweeps = 100, sweep_sequence = 'RL', verbosity = 1)
    t2_f = time.perf_counter()
    t2 = t2_f - t2_i

    if succ:

        x = dmrg.state
        exact_bds = x.bond_sizes()
        pred_bds_float = model(torch_data).detach().numpy()[0,:-1]
        pred_bds = [int(item + 3) for item in pred_bds_float]
        # set up dmrg1, dmrg_guided with exact data, and dmrg_guided with model data
        dmrg1 = qtn.DMRG1(H, bond_dims = [int(sum(exact_bds)/len(exact_bds))
                                          + ex for ex in range(20)], cutoffs=svd_cutoffs)
        dmrg_g = DMRG_guided(H, exact_bds, expand = 2, stride = 3, cutoffs = svd_cutoffs)
        dmrg_a = DMRG_guided(H, pred_bds, expand = 2, stride = 3, cutoffs = svd_cutoffs)

        # compare their convergence wall time
        t1_i = time.perf_counter()
        conv1 = dmrg1.solve(energy_tol, max_sweeps = 100, verbosity = 1)
        t1_f = time.perf_counter()
        t1 = t1_f - t1_i

        tg_i = time.perf_counter()
        convg = dmrg_g.solve(energy_tol, max_sweeps = 100, verbosity = 1)
        tg_f = time.perf_counter()
        tg = tg_f - tg_i

        ta_i = time.perf_counter()
        conva = dmrg_a.solve(energy_tol, max_sweeps = 100, verbosity = 1)
        ta_f = time.perf_counter()
        ta = ta_f - ta_i



        time_data = (t1, conv1, tg, convg, ta, conva, t2)
        fname_time = 'data/time_data_' + this_id
        np.save(fname_time, time_data)

        energy_data = (dmrg_g.energy - dmrg.energy,
                       dmrg_a.energy - dmrg.energy, dmrg1.energy - dmrg.energy)
        fname_energy = 'data/energy_data_' + this_id
        np.save(fname_energy, energy_data)


        # do compression
        #compression_tol = 1e-6

        #print('compressing...')
        #xp = compress_to_overlap_tol(x, 1e-6, None, 0)
        #print('done compressing')

        #x.compress()
        #xp.compress()

        # save data
        #run_data = np.zeros((L, 4))
        #run_data[:L-1,0] = j
        #run_data[:,1] = h
        #run_data[:L-1, 2] = x.bond_sizes()
        #run_data[:L-1, 3] = xp.bond_sizes()

        #filename = 'data/cos_random/run_data_' + this_id
        #np.save(filename, run_data)
