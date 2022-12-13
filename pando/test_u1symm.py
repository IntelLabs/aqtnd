# Unit tests for u1symm.py

import pytest

# import quimb
import quimb.tensor as qtn
from quimb.utils import oset
import numpy as np

import u1symm





def test_init_U1Tensor():
    """Test initalization of U1Tensor"""

    # ----------------------------------
    # Test 1 - rank-2 size-1x1 tensor
    inds = ('a','b')
    # ns_ds = {'a':[(1,1)], 'b':[(-1,1)]}
    ns_ds = {'a':{1:1,}, 'b':{-1:1,} }
    data = {1: [np.exp(1j*np.pi/4)]}
    try:
        testtens1 = u1symm.U1Tensor(data, inds, ns_ds)
    except Exception as exc:
        print(exc)
        assert False, f"U1Tensor.__init__() raised an exception: {exc}"
    
    # ----------------------------------
    # Test 2 - rank-2 size-4x4 tensor
    """
    |  |
    [--]
    |  |
    """
    theta = np.pi/6
    c = np.cos(theta)
    s = np.sin(theta)
    inds = ('a','b')
    # ns_ds = {'a':[(0,1),(1,2),(2,1)], 'b':[(0,1),(-1,2),(-2,1)]}
    ns_ds = {'a':{0:1, 1:2, 2:1}, 'b':{0:1, -1:2, -2:1} }
    data = {0: np.array([1.]),
            1: np.array([[c,-s],[s,c]]),
            2: np.array([1.])}
    try:
        testtens2 = u1symm.U1Tensor(data, inds, ns_ds)
    except Exception as exc:
        assert False, f"U1Tensor.__init__() raised an exception {exc}"

    # ----------------------------------
    # Test 3 - rank-3 size-4x4x4 tensor
    theta = np.pi/6
    c = np.cos(theta)
    s = np.sin(theta)
    data = {0: np.array([1.]),
            1: np.array([[c,-s],[s,c]]),
            2: np.array([1.])
    }
    inds = ('a','b','c')
    # ns_ds = {'a':[(0,1),(1,2),(2,1)], 'b':[(0,1),(-1,1)], 'c':[(0,1),(-1,1)]}
    ns_ds = {'a':{0:1, 1:2, 2:1}, 'b':{0:1, -1:1}, 'c':{0:1, -1:1} }
    try:
        testtens1 = u1symm.U1Tensor(data, inds, ns_ds)
    except Exception as exc:
        assert False, f"U1Tensor.__init__() raised an exception {exc}"

    # TODO: __init__ must have checks on shape of data
    # Should also be option to not include the data, of course.

    # ----------------------------------
    # Test 4 - Throw error: all of an index's QNs must be non-neg or non-pos
    # (or wait--is that reasonable?)
    # (maybe we shouldn't actually restrict like that)
    # (actually, i think it's easiest to restrict like this, especially
    # for first version. Can modify later. But this still covers all the 
    # cases, since a pos-and-neg index can be split into 2)


    '''
    Notes to self:
    * 'in' is positive, 'out' is negative
    '''
    
# def test_contract_U1Tensors():
#     """Test tensor_contract() for U1Tensors."""
    
#     # ----------------------------------
#     # Test 1 - rank-2 size-2x2 tensor
#     matA = np.array([[1,2],[3,4.]])
#     matB = np.array([[5,6],[7,8.]])
#     matC = matA.dot(matB)
#     inds = ('a','b')
#     ns_ds = {'a':[(1,2)], 'b':[(-1,2)]}
#     data = {1: matA,
#             }
#     t1 = u1symm.U1Tensor(data, inds, ns_ds)
#     inds = ('b','c')
#     ns_ds = {'b':[(1,2)], 'c':[(-1,2)]}
#     data = {1: matB
#             }
#     t2 = u1symm.U1Tensor(data, inds, ns_ds)
#     t_gold = u1symm.U1Tensor( {1: matA.dot(matB)} , ('a','c'), {'a':[(1,2)], 'c':[(-1,2)]})

#     t_result = qtn.tensor_contract( [t1,t2], backend=MYNEWFUNC )

#     assert t_gold==t_result




# def test_split_U1Tensors():
#     """Test split() for U1Tensors.
    
#     svd --> u-d-w
#     absorb options:
#     'left'  --> ud_w
#     'right' --> u_dw
#     'both'  --> usd_sdw (u-sqrtd_sqrtd-w)
#     """

#     # ----------------------------------
#     # Test 1 - rank-2 size-3x3 tensor
#     inds = ('a','K')
#     ns_ds = {'a':[(0,1),(1,2)], 'b':[(0,1),(-1,2)]}
#     data = {0: np.array([1.]),
#             1: np.array([[1,1],[1,-1.]])
#             }    
#     # Initialize tensor
#     t = u1symm.U1Tensor(data, inds, ns_ds)

#     # so what does the typical case return??
#     # depends what 'get' is set to. Default is get=None.
#     # assume it returns tuple[U1Tensor] of (left,right)

#     # Test 1a - 'left' with svd
#     inds = ('a','k')
#     data = {0: np.array([1.]),
#             1: np.array([
#                 [-1,-1],
#                 [-1,1.]
#             ])
#             }
#     left_gold = u1symm.U1Tensor( data,inds,ns_ds )
#     inds = ('k','b')
#     data = {0: np.array([1.]),
#             1: np.array([
#                 [-1, 0],
#                 [0 ,-1.]
#             ])
#             }
#     right_gold = u1symm.U1Tensor( data,inds,ns_ds )
#     ud_w_gold = (left_gold,right_gold)
#     ud_w_res  = t.split('a', absorb='left')
#     assert ud_w_gold==ud_w_res
#     assert ud_w_res[0].inds[0]=='a'
#     assert ud_w_res[1].inds[1]=='b'


#     # Test 1b - 'right' with svd
#     inds = ('a','k')
#     data = {0: np.array([1.]),
#             1: np.sqrt(1/2)*np.array([
#                 [-1,-1],
#                 [-1,1.]
#             ])
#             }
#     left_gold = u1symm.U1Tensor( data,inds,ns_ds )
#     inds = ('k','b')
#     data = {0: np.array([1.]),
#             1: np.sqrt(2)*np.array([
#                 [-1, 0],
#                 [0 ,-1.]
#             ])
#             }
#     right_gold = u1symm.U1Tensor( data,inds,ns_ds )
#     u_dw_gold = (left_gold,right_gold)
#     u_dw_res  = t.split('a', absorb='right')
#     assert u_dw_gold==u_dw_res
#     assert u_dw_res[0].inds[0]=='a'
#     assert u_dw_res[1].inds[1]=='b'

#     # Test 1c - 'both' with svd (Note the fourth-roots [**0.25])
#     inds = ('a','k')
#     data = {0: np.array([1.]),
#             1: (0.5**0.25) * np.sqrt(1/2)*np.array([
#                 [-1,-1],
#                 [-1,1.]
#             ])
#             }
#     left_gold = u1symm.U1Tensor( data,inds,ns_ds )
#     inds = ('k','b')
#     data = {0: np.array([1.]),
#             1: (2.**0.25)*np.array([
#                 [-1, 0],
#                 [0 ,-1.]
#             ])
#             }
#     right_gold = u1symm.U1Tensor( data,inds,ns_ds )
#     usd_sdw_gold = (left_gold,right_gold)
#     usd_sdw_res  = t.split('a', absorb='both')
#     assert usd_sdw_gold==usd_sdw_res
#     assert usd_sdw_res[0].inds[0]=='a'
#     assert usd_sdw_res[1].inds[1]=='b'


#     # Test 1d - None should throw exception for now
#     with pytest.raises(NotImplementedError):
#         t.split('a', absorb=None)




    """
    Can run tn.split() or tensor_split(tn)
    """


    """
    B = np.array([[1,1],[1,-1.]])
    u,d,v = np.linalg.svd(B)
    display(u,d,v)
    # Left:
    
    # Right:

    # Both:

    # (throw exception for 'none' since i don't want to deal w that)

    ---
    array([[-0.70710678, -0.70710678],
        [-0.70710678,  0.70710678]])
    array([1.41421356, 1.41421356])
    array([[-1., -0.],
        [-0., -1.]])
    """



    """
    TODO:
    What if indices are in a different order?
    """




def test_simple_funcs():
    """Test of simpler member function overloads"""

    pass














