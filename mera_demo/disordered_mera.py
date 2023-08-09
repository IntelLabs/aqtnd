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
from copy import copy

from warnings import simplefilter
simplefilter("ignore", category=FutureWarning)

from math import cos, sin, pi

opt = ctg.ReusableHyperOptimizer(
    progbar=True,
    reconf_opts={},
    max_repeats=16,
    # directory=  # set this for persistent cache
)

def dist_distance(idxs1, idxs2):
    dist = 0
    i2 = copy(idxs2)
    for idx in idxs1:
        if idx in i2:
            i2.pop(i2.index(idx))
        else:
            dist += 1
            
    return dist


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
    
def tag_mera(mera):
    for tag,tensor in mera.tensor_map.items():
        tensor.add_tag(str(tag))
    
def get_entanglement_spectra(mera):
    spectra = []
    unis = mera.select('_UNI').tensors
    for uni in unis:
        
        # nbs = mera.select_neighbors(uni.tags, which='all')
        # mylayer = int([x for x in uni.tags if '_LAYER' in x][0][-1])
        # upneighbors = []
        # downneighbors = []
        # for x in nbs:
        #     if '_LAYER'+str(mylayer-1) in x.tags:
        #         downneighbors.append(x)
        #     elif '_LAYER'+str(mylayer) in x.tags:
        #         upneighbors.append(x)
                
        lbs = uni.left_inds
        rbs = [x for x in uni.inds if x not in uni.left_inds]
        
        b1 = [lbs[0], rbs[0]]
        b2 = [lbs[1], rbs[1]]
        
        r = uni.to_dense(b1, b2)
        u,s,vh = np.linalg.svd(r)
        
        spectra.append((s, uni.tags))
        
    return spectra

def hist_spectra(spectra, coloridx = None):
    allsings = np.array(sum([list(x) for x,_ in spectra], []))
    plt.figure()
    plt.hist(np.log(allsings/np.max(allsings)))
    plt.figure()
    plt.semilogy(allsings/np.max(allsings), marker='x', linestyle='')
    if coloridx is not None:
        colorsings = spectra[coloridx][0]
        plt.semilogy(np.arange(coloridx * len(colorsings),(coloridx+1) * len(colorsings)),
                     colorsings/np.max(allsings), marker='x', linestyle='', color='r')
    
    
def to_truncate(spectra, method = 'svd'):
    
    sizes = [int(len(s)**(0.5)) for s,_ in spectra]
    # eligible = [x for x in sizes if x > 2]
    
    truncs = [x **2 - (x-1)**2 for x in sizes]
    
    # shouldn't truncate anything of size 2
    losses = [(sum(sorted(s)[:t])/sum(np.array(s) ** 2), x) 
              for (s,x),t,size in zip(spectra,truncs,sizes)]
    
    for k in range(len(sizes)):
        if sizes[k] < 3:
            losses[k] = (100, losses[k][1])
    
    if method == 'svd':
        idx = np.argmin([x[0] for x in losses])
    elif method == 'random':
        idx = np.random.randint(len(losses))
    elif method == 'largest':
        idx = np.argmax(sizes)
    else:
        raise ValueError('truncation method not supported')
    
    return (losses[idx], idx)

def update_mera_local(mera, localtn, unitags):
    emoc = mera.copy()
    isoneighbors = emoc.select_neighbors(unitags, which='all')
    emoc[unitags] = localtn[unitags]
    for iso in isoneighbors:
        newiso = iso.copy()
        projectors = [x for x in localtn['ADDED_PROJECTOR'] 
                          if len(x.filter_bonds(iso)[0]) > 0]
        
        if len(projectors) > 1:
            for projector in projectors:
                newiso = newiso @ projector
            newiso = qu.tensor.IsoTensor(newiso)
            newiso.left_inds = newiso.inds[:2]
        else:
            projector = projectors[0]
            lip = iso.left_inds
            li = []
            for lidx in lip:
                if lidx in projector.inds:
                    lidx = next(x for x in projector.inds if x not in iso.inds)
                li.append(lidx)
                
            newtensor = iso @ projector
            newiso = qtn.IsoTensor(newtensor)
            newiso.left_inds = li
            
        newiso.unitize_()
        emoc[iso.tags] = newiso
    return emoc
    
    
def truncate_unitary(mera, unitags, terms, method='randomize', localoptiterations = None):
    # truncate one of the unitaries of the MERA
    emoc = mera.copy()
    unitary = emoc.select(unitags).tensors[0]
    isoneighbors = emoc.select_neighbors(unitags, which='all')
    # intags = [x.tags for x in isoneighbors]
        
    unibond = unitary.shape[0] - 1
    if unibond < 1:
        return emoc

    if method == 'randomize':

        newunimat = qu.rand_iso(unibond ** 2, unibond ** 2, dtype=dtype)
        newunimat.shape = (unibond, unibond, unibond, unibond)
        newunitary = qu.tensor.IsoTensor(newunimat, inds = unitary.inds, tags = unitary.tags,
                                         left_inds = unitary.inds[:2])
        
        emoc[unitags] = newunitary
        
        for iso in isoneighbors:
            connector = iso.filter_bonds(unitary)[0][0]
            shape = list(iso.shape)
            myinds = list(iso.inds)
            cidx = myinds.index(connector)
            shape[cidx] = unibond
            
            if len(shape) == 3:
                newisomat = qu.rand_iso(shape[0] * shape[1], shape[2], dtype=dtype)
            else:
                newisomat = qu.rand_iso(shape[0], shape[1], dtype=dtype)
            newisomat.shape = shape
            newiso = qu.tensor.IsoTensor(newisomat, inds = myinds, tags = iso.tags, 
                                         left_inds = myinds[:-1])
            
            # mytid = list(emoc._get_tids_from_tags(iso.tags))[0]
            mytid = list(emoc._get_tids_from_inds(iso.inds))[0]
            # print(mytid)
            # print(emoc.tensor_map[mytid])
            emoc.tensor_map[mytid] = newiso
            # print(emoc.tensor_map[mytid])
            # print('next...')
            
            # emoc[iso.tags] = newiso
            # iso = newiso
            
        # eoptc = qtn.TNOptimizer(
        #     emoc,
        #     loss_fn=loss_energy,
        #     norm_fn=norm_fn,
        #     loss_constants={'terms': terms},
        #     loss_kwargs={'optimize': opt},
        #     autodiff_backend='torch', jit_fn=True,
        # )
        # eoptc.optimizer = 'adam'  # the default
        # emoc = eoptc.optimize(1)
    elif method == 'local':
        
        newunimat = qu.rand_iso(unibond ** 2, unibond ** 2, dtype=dtype)
        newunimat.shape = (unibond, unibond, unibond, unibond)
        newinds = [qtn.rand_uuid() for _ in range(4)]
        newunitary = qu.tensor.IsoTensor(newunimat, inds = newinds, 
                                         tags = unitary.tags, left_inds = newinds[:2])
        
        uni_inds = unitary.inds
        newuni_inds = newunitary.inds
        projpairs = zip(uni_inds, newuni_inds)
        
        projmat = np.zeros((unibond + 1, unibond), dtype=dtype)
        projmat[np.arange(unibond), np.arange(unibond)] = 1
        
        projectors = [qu.tensor.IsoTensor(data = projmat, inds = ind, 
                                       tags = 'ADDED_PROJECTOR' ,left_inds = [ind[0],])
                      for ind in projpairs]
        localtn = qtn.TensorNetwork(projectors + [newunitary])
        
        if np.abs(localtn @ localtn.H - unibond ** 2) > 0.1:
            raise ValueError('local tn is not unitary!')
        
        # if localtn @ localtn.H
        
        if localoptiterations and localoptiterations > 0:
            def local_loss(localtn, unitary):
                # print((localtn & unitary.H).contract(all))
                return -((localtn & unitary.H).contract(all))
            
            def local_energy_loss(localtn, mera, unitary, terms):
                # print((localtn & unitary.H).contract(all))
                unitags = unitary.tags
                checker = update_mera_local(mera, localtn, unitags)
                return loss_energy(checker, terms)
            
            def local_norm(localtn):
                return qtn.TensorNetwork([tt.unitize(method='exp') for tt in localtn.tensors])
                # localtn['_UNI'].unitize_(method='exp')
                # _ = [tt.unitize_() for tt in localtn['ADDED_PROJECTOR']]
            
            # ltnc = localtn.copy()
            
            localopt = qtn.TNOptimizer(
                localtn,
                loss_fn=local_energy_loss,
                norm_fn=local_norm,
                # loss_constants={'unitary': unitary},
                loss_constants={'mera':emoc, 'unitary': unitary, 'terms':terms},
                # loss_kwargs={'optimize': opt},
                autodiff_backend='torch', jit_fn=True,
            )
            localopt.optimizer = 'adam'  # the default
            localtn = localopt.optimize(localoptiterations)
            
            # print(localtn @ ltnc.H)
        
        
        
        if not (np.abs(localtn @ localtn.H - unibond ** 2) < 1e-3):
            raise ValueError('local tn is not unitary!')
        emoc = update_mera_local(emoc, localtn, unitags)
        
    # elif method == 'perturb':
        
    #     for iso in isoneighbors:
    #         connector = iso.filter_bonds(unitary)[0][0]
    #         shape = list(iso.shape)
    #         myinds = list(iso.inds)
    #         cidx = myinds.index(connector)
            
            
    #         l = iso.to_dense([x for x in myinds if x != connector], [connector])
    #         r = unitary.to_dense([connector], [x for x in unitary.inds if x != connector])
            
    #         ul, sl, vl = np.linalg.svd(l)
    #         ur, sr, vr =  np.linalg.svd(r)
            
            
            
    #         shape[cidx] = unibond
    #         newisomat = qu.rand_iso(shape[0] * shape[1], shape[2], dtype=dtype)
    #         newisomat.shape = shape
    #         newiso = qu.tensor.IsoTensor(newisomat, inds = myinds, tags = iso.tags, 
    #                                      left_inds = myinds[:2])
            
    #         emoc[iso.tags] = newiso
            
    #     eoptc = qtn.TNOptimizer(
    #         emoc,
    #         loss_fn=loss_energy,
    #         norm_fn=norm_fn,
    #         loss_constants={'terms': terms},
    #         loss_kwargs={'optimize': opt},
    #         autodiff_backend='torch', jit_fn=True,
    #     )
        # eoptc.optimizer = 'adam'  # the default
        # emoc = eoptc.optimize(1)

    return emoc
    

# use single precision for quick GPU optimization
dtype = 'float32'

sz = np.array([[ 1.0,  0],
               [ 0, -1.0]], dtype=dtype)
i2 = np.eye(2, dtype=dtype)
Z1I2 = np.kron(sz, i2)

opt = ctg.ReusableHyperOptimizer(
    progbar=True,
    reconf_opts={},
    max_repeats=16,
    # directory=  # set this for persistent cache
)

# larger system
# S = 1?
# cosine overlay


# total length (currently must be power of 2)
L = 2**5

# max bond dimension
mD = 6
xD = 5


# start with a bond dimension of 2
mera = qtn.MERA.rand(L, max_bond=mD, dtype=dtype)
tag_mera(mera)

# this is the function that projects all tensors
# with ``left_inds`` into unitary / isometric form
mera.unitize_()

disorder_strength = 2.0
j0 = 1.0
coupling_vals = np.random.normal(j0, disorder_strength, L)

H2 = qu.ham_heis(2).real.astype(dtype)
X2 = np.array([[0, 0, 0, 1],
       [0, 0, 1, 0],
       [0, 1, 0, 0],
       [1, 0, 0, 0]], dtype=dtype)

builder = qtn.SpinHam1D(S=1/2, cyclic=True)
terms = {}

HAMTYPE = 'ising'

for i in range(0, L):
    if HAMTYPE == 'ising':
        # builder[i] += coupling_vals[i], 'Z'
        builder[i, i+1] += -1.0, 'X', 'X'
        builder[i, i+1] += coupling_vals[i], 'Z', 'I'
        
        
        terms[(i, (i+1)%L)] =  -1 * X2 + coupling_vals[i] * Z1I2
        # terms[(i,)] = coupling_vals[i] * sz
            
        
    elif HAMTYPE == 'heis':
       
        builder[i, i+1] += coupling_vals[i], 'X', 'X'
        builder[i, i+1] += coupling_vals[i], 'Y', 'Y'
        builder[i, i+1] += coupling_vals[i], 'Z', 'Z'
        
        terms[(i, (i+1)%L)] =  coupling_vals[i] * H2
        
        
    elif HAMTYPE == 'heis_nn':
       
        builder[i, i+1] += coupling_vals[i], 'X', 'X'
        builder[i, i+1] += coupling_vals[i], 'Y', 'Y'
        builder[i, i+1] += coupling_vals[i], 'Z', 'Z'
        
        critical_J2 = 0.24116
        
        terms[(i, (i+1)%L)] =  coupling_vals[i] * H2
        terms[(i, (i+2)%L)] =  critical_J2 * H2
        
# builder[L, L+1] += coupling_vals[0], 'X', 'X'
# builder[L, L+1] += coupling_vals[0], 'Y', 'Y'
# builder[L, L+1] += coupling_vals[0], 'Z', 'Z'

# Htotal = builder.build_mpo(L)

x = qtn.MPS_rand_state(L, xD, cyclic=True, dtype=dtype)
x.compress()

emo = mera.copy()
eopt = qtn.TNOptimizer(
    emo,
    loss_fn=loss_energy,
    norm_fn=norm_fn,
    loss_constants={'terms': terms},
    loss_kwargs={'optimize': opt},
    autodiff_backend='torch', jit_fn=True,
)
eopt.optimizer = 'adam'  # the default
emo = eopt.optimize(1500)


optimal = emo.copy()


emo_random = emo.copy()
emo_largest = emo.copy()

# the truncation loop
numtruncs = 12
slosses = []
truncidx = []
best_energies = [loss_energy(emo, terms)]
best_ovps = []

slosses_random = []
truncidx_random = []
best_energies_random = [loss_energy(emo, terms)]
best_ovps_random = []

slosses_largest = []
truncidx_largest = []
best_energies_largest = [loss_energy(emo, terms)]
best_ovps_largest = []

# q = get_entanglement_spectra(emo)


do_global = False

for j in range(numtruncs):
    q = get_entanglement_spectra(emo)
    (sloss, truncnext), idx = to_truncate(q)
    emo = truncate_unitary(emo, truncnext, terms,  
                           method = 'local', localoptiterations = 1000)
    
    print(f'svd: {idx}')
    
    slosses.append(sloss)
    truncidx.append(list(emo._get_tids_from_tags(truncnext))[0])
    
    best_energies.append(loss_energy(emo, terms))
    
    
    # (sloss_r, truncnext_r), idx_r = to_truncate(q, method='random')
    # emo_random = truncate_unitary(emo_random, truncnext_r, terms, 
    #                               method='local', localoptiterations = 10)
    
    # slosses_random.append(sloss_r)
    # truncidx_random.append(list(emo_random._get_tids_from_tags(truncnext_r))[0])
    
    # best_energies_random.append(loss_energy(emo_random, terms))
    # # best_ovps_random.append(np.abs(emo_random.H @ opt))
    q_largest = get_entanglement_spectra(emo_largest)
    (sloss_l, truncnext_l), idx_l = to_truncate(q_largest, method = 'largest')
    emo_largest = truncate_unitary(emo_largest, truncnext_l, terms,  
                                   method='local', localoptiterations = 1000)
    
    print(f'largest: {idx_l}')
    
    slosses_largest.append(sloss_l)
    truncidx_largest.append(list(emo_largest._get_tids_from_tags(truncnext_l))[0])
    
    best_energies_largest.append(loss_energy(emo_largest, terms))
    
    # if np.abs((best_energies[-1] - best_energies[-2]) / best_energies[-2]) > 0.01:
    #     do_global = True
    
    if do_global:
        eopt = qtn.TNOptimizer(
            emo,
            loss_fn=loss_energy,
            norm_fn=norm_fn,
            loss_constants={'terms': terms},
            loss_kwargs={'optimize': opt},
            autodiff_backend='torch', jit_fn=True,
        )
        eopt.optimizer = 'adam'  # the default
        emo = eopt.optimize(1000)
        
        best_energies.append(loss_energy(emo, terms))
        
        # eopt_random = qtn.TNOptimizer(
        #     emo_random,
        #     loss_fn=loss_energy,
        #     norm_fn=norm_fn,
        #     loss_constants={'terms': terms},
        #     loss_kwargs={'optimize': opt},
        #     autodiff_backend='torch', jit_fn=True,
        # )
        # eopt_random.optimizer = 'adam'  # the default
        # emo_random = eopt_random.optimize(1000)
        
        # best_energies_random.append(loss_energy(emo_random, terms))
        
        eopt_largest = qtn.TNOptimizer(
            emo_largest,
            loss_fn=loss_energy,
            norm_fn=norm_fn,
            loss_constants={'terms': terms},
            loss_kwargs={'optimize': opt},
            autodiff_backend='torch', jit_fn=True,
        )
        eopt_largest.optimizer = 'adam'  # the default
        # emo = eopt.optimize(1)
        emo_largest = eopt_largest.optimize(1000)
        
        # slosses_largest.append(sloss_l)
        # truncidx_largest.append(list(emo_largest._get_tids_from_tags(truncnext_l))[0])
        
        best_energies_largest.append(loss_energy(emo_largest, terms))
        
        # if emo @ emo.H > 1.1:
        #     raise ValueError('mera is not unitary!')
        # if emo_largest @ emo_largest.H > 1.1:
        #     raise ValueError('mera is not unitary!')
       
        # do_global = True
        
    
    
dists = []
for k in range(numtruncs):
    dists.append(dist_distance(truncidx[:k], truncidx_largest[:k]))
    
errors = np.array(best_energies) - best_energies[0]
errors_largest = np.array(best_energies_largest) - best_energies_largest[0]


# # do the renormalization to get singlets
# def singletize_H(Htotal):
#     ...
#     # returns a list of singlet indices [..., (left, right) ,...]

# singlets = singletize_H(Htotal)

# # now expand the bond dimensions of the mera
# for left, right in singlets:
#     minuni = mera_lowest_connector_uni(mera, left, right)
#     unisize = minuni.shape[0]
#     inds = minuni.inds
#     mera.expand_bond_dimension(unisize + 1, rand_strength = 1.0, 
#                                inds_to_expand=minuni.inds, inplace=True)
#     # for k in range(len(inds)):
#     #     # len should be 4
#     #     mera.expand_bond_dimension(minuni)


