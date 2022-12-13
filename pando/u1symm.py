# Classes and functions for U(1)-preserving tensor networks


from quimb.utils import oset
import quimb.tensor as qtn
import quimb

import numpy as arrmod # for 'array module'. To prep for other array packages.
from math import prod
import copy
import itertools

# array = qtn.tensor.array

def tags_to_oset(tags):
    """Parse a ``tags`` argument into an ordered set. (pasted from tensor_core.py)
    """
    if tags is None:
        return oset()
    elif isinstance(tags, (str, int)):
        return oset((tags,))
    elif isinstance(tags, oset):
        return tags.copy()
    else:
        return oset(tags)

class U1Tensor(qtn.IsoTensor):

    def __init__(self, data, inds, ns_ds, tags=None):
        """A U(1)-preserving tensor.
        
        The main mechanical changes in U1Tensor are:
        - the data is a direct sum of matrices, not just a single matrix
        - ns_ds specifies quantum number (QN) and degeneracies of indices

        Args:
            data (dict): The data of the tensor, by pcl number.
            inds (tuple): The indices of the tensor.
            ns_ds (dict): Key is ind string, value is dict of
                    {QN: degeneracy}.
            tags (set): The tags of the tensor.

        Returns:
            U1Tensor

        """

        # Assert dict, and that at least first item is array
        assert isinstance(data, dict)
        print(next(iter(data.values())))
        print(type(next(iter(data.values()))))
        assert isinstance( next(iter(data.values())) , (arrmod.ndarray,list) )

        self._data = {}
        for key,mat in data.items():
            self._data[key] = arrmod.array(mat)
        self._inds = tags_to_oset(inds)
        self._ns_ds = copy.copy(ns_ds)
        self._tags = tags_to_oset(tags)

        # Determine and store left indices
        self._determine_left_inds()

        # Check U1 validity
        self._check_u1_validity()

        # TODO: Add additional checks on 'data' input. Should use
        # a function that's simply check_u1_validity()
        # * Full block-diag size should match
        # * Check individual block sizes

    def _determine_left_inds(self):
        """Determine the left indices of tensor, based on sign of
        quantum numbers. Positive -> left."""

        # Sort by all-non-negative and all-negative quantum numbers
        in_inds = []  # Those with pos or only-zero (atypical) QNs
        out_inds = [] # Those with negative quantum numbers
        for ind in self._inds:
            quantum_numbers = [n for n in self._ns_ds[ind].keys()]
            if all([n >= 0 for n in quantum_numbers]):
                in_inds.append(ind)
            elif all([n <= 0 for n in quantum_numbers]):
                out_inds.append(ind)
            else:
                raise Exception("Quantum numbers must be all " +
                                "non-positive or all non-negative.")
        
        # Set left indices
        self._left_inds = tags_to_oset(in_inds)


    def _check_u1_validity(self):
        """Check that data's matrices are correct dimension"""

        """Example used in comments:
        
        left_inds are
        * 'a' - [n=0,d=1], [n=1,d=2], [n=2,d=1] | n - 0,1,2 | d - 1,2,1
        * 'b' - [n=0,d=1], [n=1,d=1]            | n - 0,1   | d - 1,1
        * 'c' - [n=2,d=4], [n=3,d=4]            | n - 2,3   | d - 4,4

        In comments, by "flow number" we mean the incoming/outgoing
        integer QN. Above examples has flow nubmers of 0,1,2,3,4,5,6.
        """

        # Prep inds and quantum numbers
        right_inds = self._inds - self._left_inds # oset subtraction
        # Create list of quantum numbers. For example, if there are three left_inds,
        # Our example yields
        # left_QN_lists    == [[0,1,2],[0,1],[2,3]] and
        # left_degen_lists == [[1,2,1],[1,1],[4,4]]
        left_QN_lists  = list([ [n for n in self._ns_ds[ind].keys() ] for ind in self._left_inds])
        right_QN_lists = list([ [n for n in self._ns_ds[ind].keys() ] for ind in right_inds])
        left_degen_lists  = list([ [d for d in self._ns_ds[ind].values() ] for ind in self._left_inds])
        right_degen_lists = list([ [d for d in self._ns_ds[ind].values() ] for ind in right_inds])

        # print("*** left_QN_lists ", left_QN_lists)
        # print("*** left_degen_lists ", left_degen_lists)
        # print("*** right_QN_lists ", right_QN_lists)
        # print("*** right_degen_lists ", right_degen_lists)

        # Product of left QNs
        # Our example yields [(0,0,2), (0,0,3), ...] for l_n_listprods
        # and [(1,1,4), (1,1,4), ...] for r_n_listprods
        l_n_listprods = list(itertools.product(*left_QN_lists))
        l_degen_listprods = list(itertools.product(*left_degen_lists))
        
        # The "degeneracy sum" gives degeneracy of that "flow number"
        l_deg_sum = {}  # Hilbert sizes of each left flow number
        for n_prod,degen_prod in zip(l_n_listprods,l_degen_listprods):
            nflow = sum(n_prod)
            # Add to Hilb size of this "flow number"
            if nflow in l_deg_sum:
                l_deg_sum[nflow] += prod(degen_prod)
            else:
                l_deg_sum[nflow]  = prod(degen_prod)

        # Product of right QNs
        r_n_listprods = list(itertools.product(*right_QN_lists))
        r_degen_listprods = list(itertools.product(*right_degen_lists))

        # Degeneracy sum of right QNs
        r_deg_sum = {}  # Hilbert sizes of each right flow number
        for n_prod,degen_prod in zip(r_n_listprods,r_degen_listprods):
            nflow = - sum(n_prod)  # Negative because outgoing numbers
            if nflow in r_deg_sum:
                r_deg_sum[nflow] += prod(degen_prod)
            else:
                r_deg_sum[nflow]  = prod(degen_prod)

        # Check that matrix sizes match up
        for nflow,mat in self._data.items():
            shape = mat.shape
            if len(mat.shape)==1:
                shape = (shape[0],1)
            if nflow in l_deg_sum and nflow in r_deg_sum:
                if shape != (l_deg_sum[nflow], r_deg_sum[nflow]):
                    raise Exception(f"Matrix of flow number {str(nflow)} does " +
                    f"not have correct dimensions. shape={mat.shape}, " +
                    f"degens=>{l_deg_sum[nflow], r_deg_sum[nflow]}")
            else:
                raise Exception(f"The flow number {str(nflow)} must be present " +
                "in both left and right flow number lists.")


    def copy(self, deep=False, virtual=False):
        raise NotImplementedError()

    def _apply_function(self,fn):
        raise NotImplementedError()

    def modify(self, **kwargs):
        raise NotImplementedError()

    def get_params(self):
        raise NotImplementedError()

    def set_params(self, params):
        raise NotImplementedError()

    def isel(self, selectors, inplace=False):
        raise NotImplementedError()

    def expand_ind(self, ind, size):
        raise NotImplementedError()

    def new_ind(self, name, size=1, axis=0):
        raise NotImplementedError()

    def new_ind_with_identity(self, name, left_inds, right_inds, axis=0):
        raise NotImplementedError()

    def conj(self):
        raise NotImplementedError()

    def ind_size(self):
        raise NotImplementedError()
    
    def shared_bond_size(self,other):
        raise NotImplementedError()

    def transpose(self, *output_inds, inplace=False):
        raise NotImplementedError()

    def moveindex(self, ind, axis, inplace=False):
        raise NotImplementedError()

    def trace(self, left_inds, right_inds, preserve_tensor=False, inplace=False):
        raise NotImplementedError()

    def sum_reduce(self, ind, inplace=False):
        raise NotImplementedError()

    def singular_values(self, left_inds, method='svd'):
        raise NotImplementedError()

    def reindex(self, index_map, inplace=False):
        raise NotImplementedError()
    
    def fuse(self, fuse_map, inplace=False):   # ***************
        raise NotImplementedError()

    def unfuse(self, unfuse_map, shape_map, inplace=False):
        raise NotImplementedError()

    def largest_element(self):
        raise NotImplementedError()

    def normalize(self, inplace=False):
        raise NotImplementedError()

    def symmetrize(self, ind1, ind2, inplace=False):
        raise NotImplementedError()

    def unitize(self, left_inds=None, inplace=False, method='qr'):
        raise NotImplementedError()

    def almost_equals(self, other, **kwargs):
        raise NotImplementedError()

    def __mul__(self, other):
        raise NotImplementedError()

    def __imul__(self, other):
        raise NotImplementedError()

    def __itruediv__(self, other):
        raise NotImplementedError()

    def __and__(self, other):
        raise NotImplementedError()

    def __or__(self, other):
        raise NotImplementedError()

    def __matmul__(self, other):
        raise NotImplementedError()

    def __repr__(self):
        return f"U1Tensor({self.data}, {self.inds}, {self.ns_ds}, {self.tags})"

    def __str__(self):
        return f"U1Tensor({self.data}, {self.inds}, {self.ns_ds}, {self.tags})"




# Overloading looks like a pain, as this is a standalone function. Might have to do a differently-named function.
# New function can then be placed in U1TensorNetwork class though.
# *** oh. might be able to *only* change the backend. ***
def tensor_split(self):
    raise NotImplementedError()


# class U1TensorNetwork( ts, *, virtual=False, check_collisions=True )
#     """
#     Notes:
#     - Indices' ns_ds must match
#     """

#     def __init__(self, tensors, check_collisions=True):

#         super().__init__(tensors, check_collisions)







