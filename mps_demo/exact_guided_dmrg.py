# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 15:57:51 2022

@author: mthibode
"""


import numpy as np
import quimb as qu
import quimb.tensor as qtn
import itertools

class DMRG_guided(qtn.DMRG):
    
    def __init__(self, ham, bond_dims, which='SA', cutoffs=1e-8, p0=None, expand = 0, stride = 1):

        super().__init__(ham, bond_dims=max(bond_dims), cutoffs=cutoffs,
                         which=which, p0=p0, bsz=1)
        for k in range(self.L - 1):
            self._k.compress_between(k, k+1, max_bond=bond_dims[k], cutoff = None)
            self._b.compress_between(k, k+1, max_bond=bond_dims[k], cutoff = None)
            
        self._expand_by = expand
        self.stride = stride
        
    def solve(self,
          tol=1e-4,
          bond_dims=None,
          cutoffs=None,
          sweep_sequence=None,
          max_sweeps=10,
          verbosity=0):
        """Solve the system with a sequence of sweeps, up to a certain
        absolute tolerance in the energy or maximum number of sweeps.

        Parameters
        ----------
        tol : float, optional
            The absolute tolerance to converge energy to.
        bond_dims : int or sequence of int
            Overide the initial/current bond_dim sequence.
        cutoffs : float of sequence of float
            Overide the initial/current cutoff sequence.
        sweep_sequence : str, optional
            String made of 'L' and 'R' defining the sweep sequence, e.g 'RRL'.
            The sequence will be repeated until ``max_sweeps`` is reached.
        max_sweeps : int, optional
            The maximum number of sweeps to perform.
        verbosity : {0, 1, 2}, optional
            How much information to print about progress.

        Returns
        -------
        converged : bool
            Whether the algorithm has converged to ``tol`` yet.
        """
        verbosity = int(verbosity)

        # Possibly overide the default bond dimension, cutoff, LR sequences.
        if bond_dims is not None:
            self._set_bond_dim_seq(bond_dims)
        if cutoffs is not None:
            self._set_cutoff_seq(cutoffs)
        if sweep_sequence is None:
            sweep_sequence = self.opts['default_sweep_sequence']

        RLs = itertools.cycle(sweep_sequence)
        previous_LR = '0'

        for sweepnum in range(max_sweeps):
            # Get the next direction, bond dimension and cutoff
            LR, bd, ctf = next(RLs), next(self._bond_dims), next(self._cutoffs)
            self._print_pre_sweep(len(self.energies), LR,
                                  bd, ctf, verbosity=verbosity)
            
            print(bond_dims)
        

            # if last sweep was in opposite direction no need to canonize
            canonize = False if LR + previous_LR in {'LR', 'RL'} else True
            # need to manually expand bond dimension for DMRG1
            
            stride = self.stride
            if stride < 1:
                stride = 1
            if self.bsz == 1 and sweepnum % stride == 0:
                expansion = self._expand_by
                for j in range(self.L - 1):
                    this_bond = self._k.bond(j, j+1)
                    # new_dim = self._k[j].ind_size(this_bond) + expansion
                    for idx in (j,j+1):
                        # self._k[idx].expand_ind(this_bond, self._k[idx].ind_size(this_bond)
                        #                   + expansion)
                        pads = [
                            (0, 0) if ind != this_bond else
                            (0, expansion)
                            for d, ind in zip(self._k[idx].shape, self._k[idx].inds)
                        ]

                        newdata = np.pad(self._k[idx].data, pads, mode = qu.tensor.tensor_core.rand_padder, 
                                         rand_strength = self.opts['bond_expand_rand_strength'])
                        self._k[idx].modify(data = newdata)
                        self._b[idx].modify(data = newdata)
                # pass
                # self._k.expand_bond_dimension(
                #     bd, bra=self._b,
                #     rand_strength=self.opts['bond_expand_rand_strength'])

            # inject all options and defaults
            sweep_opts = {
                'canonize': canonize,
                'max_bond': bd,
                'cutoff': ctf,
                'cutoff_mode': self.opts['bond_compress_cutoff_mode'],
                'method': self.opts['bond_compress_method'],
                'verbosity': verbosity,
            }

            # perform sweep, any plugin computations
            self.energies.append(self.sweep(direction=LR, **sweep_opts))
            self._compute_post_sweep()

            # check convergence
            converged = self._check_convergence(tol)
            self._print_post_sweep(converged, verbosity=verbosity)
            if converged:
                break

            previous_LR = LR

        return converged
            
    
        

SVD_CUTOFF = 1e-10

def bd_guided_dmrg(H, bd_guide):
    """
    Attempts to perform DMRG on H with starting bond dimensions given
    by bd_guide

    Parameters
    ----------
    H : quimb MPO
    bd_guide : list
        bond dimensions at which to perform DMRG

    Returns
    -------
    None.

    """
    
    dmrg_g = DMRG_guided(H, bd_guide, cutoffs=SVD_CUTOFF)
