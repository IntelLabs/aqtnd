# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 15:38:58 2022

@author: mthibode
"""

from copy import copy
import itertools
import re
from random import choice
import numpy as np
import quimb as qu
import quimb.tensor as qtn


class TTN(qtn.TensorNetwork):

    def __init__(self,  ts, parent_nodes, *, virtual=False, check_collisions=True):

        super().__init__(ts, virtual = virtual, check_collisions = check_collisions)
        # now overlay the tree structure defined by parent_nodes
        # WARNING: does not check for validity of the tree structure
        self._parents = parent_nodes
        self._canonization_center = None
        self._node_degrees = None
        self._groups = None

    def __copy__(self):

        newTN = super().copy(deep=True)
        newTN._parents = copy(self._parents)
        return newTN

    def copy(self):
        return self.__copy__()


    @classmethod
    def random_TTN(cls, leaves, phys_dim, bd_min, bd_max, degree_min = 3, degree_max = 3):
        # creates a random TTN

        # first, create the topology
        parent_nodes, node_degrees, groups = cls._rand_tree_parents(leaves, degree_min, degree_max)
        num_parents = len(node_degrees) - leaves
        # now create random tensors

        bds = {k: [[np.random.randint(bd_min, bd_max + 1),1] for _ in range(node_degrees[k])]
                      for k in range(num_parents + leaves)}

        for k in range(leaves):
            bds[k][0][0] = phys_dim
            for bond in bds[k]:
                bond[1] = 0

        for c,p in parent_nodes.items():
            if p is not None:
                cn, pn = cls.get_tree_num(c), cls.get_tree_num(p)

                cbonds = bds[cn]
                pbonds = bds[pn]

                cbond = cbonds[-1]
                pbond = next(x for x in pbonds if x[1] == 1)

                # print(cbond, pbond)

                this_bd = cbond[0]
                pbond[0] = this_bd

                # set flags
                cbond[1] = 0
                pbond[1] = 0

                # print(cbond, pbond)

        bds = {k: [x[0] for x in v] for k,v in bds.items()}

        leaf_tensors = []
        for d in range(leaves):
            # pnum = parent_nodes[f'tree{d}'].split('tree')[1]
            leaf_tensors.append(qtn.Tensor(qu.randn((phys_dim, bds[d][-1])), inds=(f'p{d}',
                                    'dummy'),
                                   tags = f'tree{d}'))
        parent_tensors = [qtn.Tensor(qu.randn(bds[d + leaves]),
                                     inds = [f'dummy{k}' for k in range(node_degrees[d + leaves])],
                                     tags=f'tree{d + leaves}')
                          for d in range(num_parents)]
        tensors = leaf_tensors + parent_tensors
        children = leaf_tensors

        while len(children) > 0:
            parents = []
            # print(children)
            # print(len(children))
            for leaf_tensor in children:
                cl = cls.get_tree_tag(leaf_tensor)
                cind = cl.split('tree')[1]
                pl = parent_nodes[cl]
                if pl is not None:
                    pind = int(pl.split('tree')[1])
                    pt = tensors[pind]

                    # print(pt)


                    cbond = next(x for x in leaf_tensor.inds if 'dummy' in x)
                    pbond = next(x for x in pt.inds if 'dummy' in x)
                    newbond = f'{cind}b{pind}'

                    leaf_tensor.reindex({cbond:newbond}, inplace=True)
                    pt.reindex({pbond:newbond}, inplace=True)
                    if pt not in parents:
                        parents.append(pt)
            # print('p' + str(len(parents)))
            children = parents


        x = TTN(tensors, parent_nodes)
        x._node_degrees = node_degrees
        x._groups = groups

        x.normalize()
        x.condition_tree()

        return x


    @staticmethod
    def _rand_tree_groups(num_leaves, dmin, dmax):
        groups = []
        # add the first layer to the tree
        layer = []
        remaining_heads = num_leaves
        while remaining_heads > 1:
            num_layer = remaining_heads
            layer = []
            while sum(layer) < num_layer:
                layer.append(np.random.randint(dmin - 1, dmax))
            # fix the last slot
            layer[-1] = num_layer - sum(layer[:-1])
            groups.append(copy(layer))

            remaining_heads = len(layer)

        return groups

    @classmethod
    def _rand_tree_parents(cls, num_leaves, dmin, dmax):
        groups = cls._rand_tree_groups(num_leaves, dmin, dmax)
        # print(groups)
        parent_dict = {}

        child_idx = 0
        parent_idx = num_leaves

        # leaves have one phyiscal bond and one virtual bond
        # virtual bond will be counted later
        node_degrees = [1] * num_leaves

        for layergroup in groups:

            # parent_idx += sum(layergroup)
            node_degrees.extend([0] * (parent_idx - child_idx))
            for step in layergroup:
                for _ in range(step):
                    cn = f'tree{child_idx}'
                    pn = f'tree{parent_idx}'
                    parent_dict[cn] = pn
                    node_degrees[parent_idx] += 1
                    node_degrees[child_idx] += 1
                    child_idx += 1
                parent_idx += 1
        ln = f'tree{parent_idx - 1}'
        parent_dict[ln] = None

        node_degrees = [x for x in node_degrees if x != 0]
        return parent_dict, node_degrees, groups

    @staticmethod
    def get_tree_tag(tensor):
        if tensor is None:
            return None
        else:
            return next(x for x in tensor.tags if 'tree' in x)

    @classmethod
    def tree_name(cls, tensor):
        if isinstance(tensor, str):
            return tensor
        else:
            return cls.get_tree_tag(tensor)

    @classmethod
    def get_tree_num(cls, tensor):
        tt = cls.tree_name(tensor)
        return int(tt.split('tree')[1])

    def next_tree_num(self):
        tree_nums = [self.get_tree_num(x) for x in self.tensors]
        return max(tree_nums) + 1

    def next_tree_tag(self):
        return 'tree' + str(self.next_tree_num())

    def retag(self, retagging):
        # does inplace only
        super().retag(retagging, inplace = True)

        retagging[None] = None
        pdict = self._parents
        newpdict = {retagging[k]: retagging[v] for (k,v) in pdict.items()}
        self._parents = newpdict
        self._canonization_center = retagging[self._canonization_center]

    def canonize_labels(self):

        retagging = {}
        nodequeue = list(self.get_leaves())
        seen = []

        counter = 0

        while len(nodequeue) > 0:
            tag = nodequeue.pop(0)
            # print(tag)
            if tag in seen or tag is None:
                continue
            else:
                seen.append(tag)
                parent = self.get_parent(tag)
                nodequeue.append(parent)
                retagging[tag] = f'tree{counter}'
                counter += 1
                # print(tag, counter)


        self.retag(retagging)


    def get_adjacency_matrix(self):
        self.canonize_labels()
        nt = self.num_tensors
        adj = np.zeros((nt, nt), dtype=int)
        for ctag,ptag in self._parents.items():
            if ptag is not None:
                c, p = self[ctag], self[ptag]
                # print(c, p)
                cn = self.get_tree_num(c)
                pn = self.get_tree_num(p)

                weight = c.ind_size(list(c.bonds(p))[0])
                adj[cn, pn] = weight
                adj[pn, cn] = weight

        return adj

    def get_parent(self, tensor):
        tree_tag = self.tree_name(tensor)
        if tree_tag not in self._parents.keys():
            return None
        else:
            return self._parents[tree_tag]

    def set_parent(self, tensor, parent_lbl):
        tree_tag = self.tree_name(tensor)
        self._parents[tree_tag] = parent_lbl


    def get_all_parents(self, tensor):
        nt = self.get_parent(tensor)
        pl = []
        while nt is not None:
            pl.append(nt)
            nt = self.get_parent(pl[-1])
        return pl

    def get_children(self, tensor):
        tree_tag = self.tree_name(tensor)
        return (k for k,v in self._parents.items() if v == tree_tag)

    def get_all_children(self, tensor):
        tree_tag = self.tree_name(tensor)
        children = list(self.get_children(tree_tag))
        for x in self.get_children(tree_tag):
            children.extend(list(self.get_all_children(x)))
        return children

    def get_descendents(self, tensor):

        tt = self.tree_name(tensor)
        children = list(self.get_children(tensor))
        if len(children) == 0:
            return [tt]
        else:
            return list(itertools.chain(*[self.get_descendents(c) for c in children]))

    def move_one_link(self, tensor, old_parent, new_parent, max_bond = None):
        # moves tensor from parent tensor old_parent to new_parent
        op = self[old_parent]
        newp = self[new_parent]
        child_link = op.filter_bonds(tensor)[0][0]
        parent_link = op.filter_bonds(newp)[0][0]
        q,r = op.split((child_link, parent_link), absorb='left', max_bond = max_bond)
        q.drop_tags(self.get_tree_tag(op))

        # new structure: old_parent --> r, new_parent --> new_parent @ q
        self[old_parent] = r
        self[new_parent] = self[new_parent] @ q

        self.set_parent(tensor, new_parent)

    def fuse_subtrees(self, root1, root2, max_bond = None):
        # fuse the subtrees rooted at root1 and root2
        # root1 and root2 should have a common parent

        pn = self.get_parent(root1)
        if pn != self.get_parent(root2):
            raise Exception('subtree roots have different parents')
        pn_t = self[pn]
        root1_t, root2_t = self[root1], self[root2]
        l1, l2 = pn_t.filter_bonds(root1_t)[0][0], pn_t.filter_bonds(root2_t)[0][0]
        pn_newleaf, pn_newhead = pn_t.split((l1,l2), max_bond=max_bond)

        pn_newleaf.drop_tags(pn)
        newtag = self.next_tree_tag()
        pn_newleaf.add_tag(newtag)

        # modify old parent
        self[pn] = pn_newhead
        self.add(pn_newleaf)
        self.set_parent(root1, newtag)
        self.set_parent(root2, newtag)
        self.set_parent(newtag, pn)

        self._canonization_center = None

    def canonize_between(self, t1, t2, absorb = 'right', max_bond = None):

        # try:
        t1t = self[t1]
        t2t = self[t2]
        _, li = t1t.filter_bonds(t2t)
        # except:
        #     self.draw(color=(t1, t2))
        #     raise Exception(f'problem with bonds: {t1}, {t2}, {self._parents}')
        if absorb is not None:
            # print(max_bond)
            q,r = t1t.split(li, absorb = absorb, max_bond = max_bond)
            # print(q.shape)
            # print(r.shape)
            s = None
        else:
            q,s,r = t1t.split(li, method = 'svd', absorb = None, max_bond = max_bond)
        r.drop_tags(q.tags)
        self[t1] = q
        self[t2] = t2t @ r

        self._canonization_center = None

        return s

    def canonize_subtree(self, parent, up = False, max_bond = None):

        children = list(self.get_children(parent))
        for child in children:
            self.canonize_subtree(child, up = True, max_bond = max_bond)

        if up:
            gp = self._parents[parent]
            self.canonize_between(parent, gp, max_bond = max_bond)

        self._canonization_center = None

    def canonize_top(self, max_bond = None):
        parent = self.get_top_parent()
        if self._canonization_center is not None:
            self.canonize_path(self._canonization_center, parent, max_bond)
        else:
            if parent is None:
                raise Exception('error finding top parent')
            self.canonize_subtree(parent, False, max_bond)
        self._canonization_center = parent

    def canonize_path(self, t1, t2, max_bond = None):

        path = self.get_path(t1, t2)[1:]
        ct = t1
        for nt in path:
            self.canonize_between(ct, nt, max_bond = max_bond)
            ct = nt


        self._canonization_center = None

    def canonize_bond(self, child, parent, get = False):

        self.canonize_top()
        self.canonize_path(self.get_top_parent(), parent)
        if get:
            s = self.canonize_between(child, parent, absorb = None)
        else:
            self.canonize_between(child, parent)
            s = None
        self._canonization_center = None
        return s

    def canonize_site(self, site, max_bond = None):
        if self._canonization_center is None:
            self.canonize_top(max_bond)
        self.canonize_path(self._canonization_center, site, max_bond)
        self._canonization_center = site

    def normalize(self):

        self.canonize_top()
        tidx = self.get_top_parent()
        top = self[tidx]
        u,_,v = top.split(top.inds[0], absorb=None)
        nt = u @ v
        self[tidx] = nt / np.sqrt(nt @ nt.H)


    def get_leaves(self):
        children, parents = zip(*self._parents.items())
        return list(sorted(x for x in children if x not in parents))

    def get_top_parent(self):
        children, parents = zip(*self._parents.items())
        return next(x for x in parents if (x not in children
                    or self._parents[x] is None) and x is not None)


    def get_path(self, t1, t2):
        # return the shortest path from t1 to t2 as an (ordered) list
        # of tensor labels, including t1 and t2
        t1p = [t1] + self.get_all_parents(t1)
        t2_parents = [t2]
        while t2_parents[-1] not in t1p:
            t2_parents.append(self.get_parent(t2_parents[-1]))
        t1p_endidx = t1p.index(t2_parents[-1])
        return t1p[:t1p_endidx] + list(reversed(t2_parents))


    def transport_subtree(self, root, destination, max_bond = None):
        # transports the subtree rooted at the tensor root though ttn so that it
        # connects to the tensor destination

        idxlist = self.get_path(root, destination)[1:] # exclude root from idxlist
        parent_tensor_label = idxlist.pop(0)
        for tensor in idxlist:
            self.move_one_link(self[root], parent_tensor_label, tensor, max_bond)
            parent_tensor_label = copy(tensor)

    def condition_subtree(self, root):
        # contract any nodes in the subtree with only two bound edges

        children = list(self.get_children(root))
        for child in children:
            # print(child)
            self.condition_subtree(child)

        if len(children) == 1:

            child = children[0]
            # self.draw(color=(root, child))
            ct = self[child]
            self[root] = self[root] @ ct
            self[root].drop_tags(ct.tags)
            gcs = self.get_children(child)
            for gc in gcs:
                self._parents[gc] = root
            # _ = self._parents.pop(child)
            self.delete(child)

            # print('deleted ' + child)


    def condition_tree(self):
        self.condition_subtree(self.get_top_parent())


    def rename_phys_indices(self, prefix):
        if len(prefix) != 1:
            raise Exception('prefixes must be exactly one character')
        for leaf in self.get_leaves():
            _, pli = self[leaf].filter_bonds(self[self._parents[leaf]])
            for b in pli:
                self[leaf].reindex_({b: prefix + b[1:]})

    def bra(self):

        xb = self.H.copy()
        xb.rename_phys_indices('b')
        return xb

    def ket(self):

        xk = self.copy()
        xk.rename_phys_indices('k')
        return xk


    def delete(self, tag):
        # print(self._parents)
        # print(tag)
        super().delete(tag)
        self._parents.pop(tag)


def energy(tree, mpo):
    return tree.bra() & mpo & tree.ket()


def local_ham(mpo, ttn, site, max_bond, max_coord = 3):

    mpo.add_tag('_HAM')
    ttn.canonize_site(site)
    x = ttn.copy()
    # x.canonize_site(site)
    for tensor in x.tensors:
        tensor.add_tag('_HAM')
    x[site].drop_tags('_HAM')
    # x.delete(site)
    xb = x.bra()
    xk = x.ket()
    xb.mangle_inner_(append='bra')
    xk.mangle_inner_(append='ket')

    network = (xb & mpo & xk)
    untagged_size = 1
    for t in network.tensors:
        if '_HAM' not in t.tags:
            untagged_size *= np.prod(t.shape)

    if untagged_size > max_bond ** (max_coord * 2):
        raise Exception(f'tensor too big: {untagged_size}, {site}, {ttn[site]}')
    y = ((xb | mpo | xk)^ '_HAM')['_HAM']
    # y = (xb & mpo & xk) ^ ...



    ket_phys = [x for x in y.inds if re.match(r'k\d+', x)]
    # ket_internal = [x for x in y.inds if re.match('.*ket', x)]
    bra_phys = [x for x in y.inds if re.match(r'b\d+', x)]
    # bra_internal = [x for x in y.inds if re.match('.*bra', x)]

    # match up the indices properly
    ki = list(sorted(ket_phys))
    bi = list(sorted(bra_phys))
    xi = copy(ki)

    for nt in x.select_neighbors(site):
        st = nt.tags
        ki.append(list(xk[site].bonds(xk[st]))[0])
        bi.append(list(xb[site].bonds(xb[st]))[0])
        xi.append(list(x[site].bonds(x[st]))[0])

    yham = y.to_dense(bi, ki)

    return y, yham, ki, bi, xi

def sweep_subtree(ttn, mpo, L, max_bond, top, reps = 3, reverse = True, method='vertical'):
    # top = ttn.get_top_parent()

    energies = [np.real(energy(ttn, mpo)^...)]
    sweepleaves = ttn.get_descendents(top)#list(ttn.get_leaves())

    reverse_round = False
    for _ in range(reps):
        if method == 'vertical':

            for start in sweepleaves:
                path = ttn.get_path(start, top)
                for tag in path[:]:
                    try:
                        ttn.canonize_site(tag, max_bond)
                        h, hm, _, _, _ = local_ham(mpo, ttn, tag, max_bond)
                        # print(hm.shape)
                        _,v = np.linalg.eigh(hm)
                        if np.max(np.abs((hm - hm.conj().T))) > 1e-11:
                            print( np.max(np.abs((hm - hm.conj().T))))
                            raise Exception('not hermitian!')
                        v0 = v[:,0].reshape(ttn[tag].shape)#h.shape[:len(h.shape)//2])

                        ttn[tag].modify(data = v0)
                        energies.append(np.real(energy(ttn, mpo) ^ ...))
                    except:
                        print(tag)
                        print(v[:,0].shape)
                        print(ttn[tag].shape)
                        print(h.shape)
                        # print(h.shape[:len(h.shape)//2])
                        raise Exception('dmrg failed')
                    # print(f'delta E: {energies[-1] - energies[-2]}')
            if reverse:
                sweepleaves = list(reversed(sweepleaves))

        elif method == 'horizontal':
            stopflag = False
            while not stopflag:
                next_layer = []
                for start in sweepleaves:
                    ttn.canonize_site(start, max_bond)
                    h, hm, ki, bi, xi = local_ham(mpo, ttn, start, max_bond)
                    w,v = np.linalg.eigh(hm)
                    if np.max(np.abs((hm - hm.conj().T))) > 1e-11:
                        print( np.max(np.abs((hm - hm.conj().T))))
                        raise Exception('not hermitian!')
                    v0 = v[:,0].reshape(ttn[start].shape)
                    ttn[start].modify(data = v0)
                    energies.append(np.real(energy(ttn, mpo) ^ ...))
                    # print(f'delta E: {energies[-1] - energies[-2]}')

                    if reverse_round:
                        plist = ttn.get_children(start)
                    else:
                        plist = [ttn.get_parent(start)]

                    for p in plist:
                        if p not in next_layer and p is not None:
                            next_layer.append(p)
                if len(next_layer) == 0:
                    stopflag = True
                else:
                    sweepleaves = copy(next_layer)
                # print(sweepleaves)

        if reverse:
            reverse_round = not reverse_round


        # print(f'swept {k}')

    return energies

def sweep_tree(ttn, mpo, L, max_bond, reps = 3, reverse = True, method='vertical'):

    ttn.canonize_top(max_bond)
    return sweep_subtree(ttn, mpo, L, max_bond, ttn.get_top_parent(), reps, reverse, method)

def evaluate_move(new_energy, prior_energy, T):

    x = np.random.rand(1)[0]

    return bool(x < np.exp(-(new_energy - prior_energy)/T))


def propose_move(ttn, max_bond, movetype='shift_split'):

    leaves = list(ttn.get_leaves())
    startnodes = [ttn.get_tree_tag(x) for x in ttn.tensors
                  if ttn.get_tree_tag(x) != ttn.get_top_parent()]
    endnodes = [x for x in startnodes if x not in leaves]
    # print(startnodes, endnodes, leaves)

    ttn_c = ttn.copy()

    if movetype == 'shift_split':
        # preserves coordination number of the TTN

        start = choice(startnodes)
        if start in endnodes:
            endnodes.pop(endnodes.index(start))
        parent = ttn_c.get_parent(start)
        if parent in endnodes:
            endnodes.pop(endnodes.index(parent))
        children = ttn_c.get_all_children(start)
        for child in children:
            if child in endnodes:
                endnodes.pop(endnodes.index(child))

        old_parent = ttn_c.get_parent(start)

        if len(endnodes) > 0:
            end = choice(endnodes)

            print(start, end)

            children = list(ttn_c.get_children(end))

            ttn_c.transport_subtree(start, end, max_bond)

            if len(children) > 0:
                end_fuse = choice(children)
                print(end_fuse)
                # ttn_c.draw(color=(start, end_fuse))
                ttn_c.fuse_subtrees(start, end_fuse, max_bond)
                # ttn_c.get_top_parent()

            # ttn.canonize_top()
        else:
            end = old_parent
            end_fuse = start

    new_parent = ttn_c.get_parent(end_fuse)
    moved_node = start
    return ttn_c, moved_node, old_parent, new_parent

def anneal_timestep(ttn, mpo, L, T, max_bond, rounds = 100, method='vertical'):

    state = ttn.copy()
    energies = [sweep_tree(state, mpo, L, max_bond, reps = 2, method=method)[-1]]
    states = [state.copy()]
    all_energies = copy(energies)
    for _ in range(rounds):
        x, _, _, _ = propose_move(state, max_bond)
        # e = sweep_subtree(x, mpo, L, max_bond, old_parent, reps = 1, method='vertical')[-1]
        # e = sweep_subtree(x, mpo, L, max_bond, new_parent, reps = 1, method='vertical')[-1]
        e = sweep_tree(x, mpo, L, max_bond, reps = 1, method=method)[-1]

        all_energies.append(e)

        x.condition_tree()
        keep = evaluate_move(e, energies[-1], T)

        if keep:
            print('nice')
            state = x
            states.append(x.copy())
            energies.append(e)

        else:
            print('rip')

    return states, energies, all_energies



def optimize_MPO(H, max_bond, rounds = 10, min_coord = 3, max_coord = 3):

    H.add_tag('_HAM')

    L = H.L
    x = TTN.random_TTN(L, 2, max_bond, max_bond, min_coord, max_coord)

    #init_e = energy(x,H) ^ ...
    init_e = sweep_tree(x, H, L, max_bond, reps = 2)

    temp = 1e-2
    states, energies, all_energies = anneal_timestep(x, H, L, temp,
                                                     max_bond, rounds, method='vertical')



    return states, energies, all_energies, init_e


def minimize_stream(series, optim_func = min):
    return [optim_func(series[:k+1]) for k in range(len(series))]
