# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 11:39:33 2022

@author: mthibode
"""

# import numpy as np
import re
from math import log2
# import quimb as qu
import quimb.tensor as qtn

class TruncMERA(qtn.TensorNetwork):

    def __init__(self, L_lower, L_upper, mera,
                 lower_site_ind_id = 'k{}', upper_site_ind_id = 'l{}'):

        # mera = qtn.MERA(L_lower, uni = uni, iso = iso, phys_dim = phys_dim, dangle = dangle,
                 # site_ind_id=site_ind_id, site_tag_id=site_tag_id, **tn_opts)

        n_layers = round(log2(L_lower) - log2(L_upper))
        layertags = [f'_LAYER{j}' for j in range(n_layers)]

        super().__init__(mera.select(layertags, which='any'))

        # print(lower_site_ind_id[:-2] + '\d+')
        upper_inds = [x for x in self.outer_inds()
                      if not re.match(lower_site_ind_id[:-2] + r'\d+', x)]
        # print(upper_inds)
        uimap = {upper_inds[j]: upper_site_ind_id[:-2] + str(j) for j in range(len(upper_inds))}
        self.reindex(uimap, inplace = True)





    @classmethod
    def rand(cls,  L_lower, L_upper, max_bond=None, phys_dim=2, dtype=float,
             lower_site_ind_id = 'k{}', upper_site_ind_id = 'l{}', **mera_opts):
        mera_opts['site_ind_id'] = lower_site_ind_id
        mera = qtn.MERA.rand(L_lower, max_bond = max_bond,
                             phys_dim = phys_dim, dtype=dtype, **mera_opts)
        return cls(L_lower, L_upper, mera, lower_site_ind_id, upper_site_ind_id)
