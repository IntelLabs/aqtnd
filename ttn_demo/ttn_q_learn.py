# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 17:12:10 2022

@author: mthibode
"""


from random import random, choice
from copy import copy
import numpy as np
# import torch
# import quimb
# import quimb.tensor as qtn
from ttn import sweep_tree



def eps_greedy(move_reward_dict, epsilon):

    k, rewards = list(zip(*move_reward_dict.items()))
    _, moves = zip(*k)
    best_move = moves[np.argmax(rewards)]
    random_move = choice(moves)

    roll = random()
    if roll > 1 - epsilon:
        move = random_move
    else:
        move = best_move

    return move


def generate_all_moves(ttn):

    # move consists of a 3-tuple (child, new_grandparent, new_sibling) where first child is
    # moved to become a child of new_grandparent and then fused with new_sibling

    # ttn labels should have been canonized by calling .canonize_labels()

    leaves = list(ttn.get_leaves())
    startnodes = [ttn.get_tree_tag(x) for x in ttn.tensors
                  if ttn.get_tree_tag(x) != ttn.get_top_parent()]
    endnodes_master = [x for x in startnodes if x not in leaves]
    # print(startnodes, endnodes, leaves)

    moves = []

    for start in startnodes:
        endnodes = copy(endnodes_master)
        if start in endnodes:
            endnodes.pop(endnodes.index(start))
        parent = ttn.get_parent(start)
        if parent in endnodes:
            endnodes.pop(endnodes.index(parent))
        children = ttn.get_all_children(start)
        for child in children:
            if child in endnodes:
                endnodes.pop(endnodes.index(child))

        for end in endnodes:

            children = list(ttn.get_children(end))

            for end_fuse in children:
                moves.append((start, end, end_fuse))

    return moves

def execute_move(ttn, move, max_bond):
    start, end, end_fuse = move

    ttn.transport_subtree(start, end, max_bond)

    ttn.fuse_subtrees(start, end_fuse, max_bond)

    ttn.condition_tree()

def get_state(ttn):
    ttn.canonize_labels()
    adj = ttn.get_adjacency_matrix()
    # only care about topology for now, not edge weights
    adj[adj > 0] = 1
    return tuple(tuple(x) for x in adj)

def get_reward(ttn, mpo, L, max_bond):
    energies = sweep_tree(ttn, mpo, L, max_bond, reps = 2)
    return -1 * energies[-1]

def look_ahead(Qdict, ttn):
    thisQdict = restrict_move_reward_dict(Qdict, ttn)
    _, rewards = zip(*thisQdict.items())
    return max(rewards)


def init_move_reward_dict(ttn, init_val = 0):

    ttn_state = get_state(ttn)
    moves = generate_all_moves(ttn)
    return {(ttn_state, x): init_val for x in moves}

def extend_move_reward_dict(Qdict, ttn, init_val = 0):
    newmoves = init_move_reward_dict(ttn, init_val)
    newmoves.update(Qdict)

    return newmoves

def restrict_move_reward_dict(Qdict, ttn):

    state = get_state(ttn)
    return {k:v for (k,v) in Qdict.items() if k[0] == state}


def q_learn(ttn, mpo, max_bond, num_episodes, episode_length, discount = 0.3,
            learning_rate = 0.2, epsilon = 0.1, Qinit = None):

    L = mpo.L
    best_Rs = [0]
    improved_steps = [0]
    best_state = ttn.copy()

    if Qinit is None:
        Qdict = init_move_reward_dict(ttn)
    else:
        Qdict = Qinit

    # print('obtained Q dict')


    for j in range(num_episodes):

        for k in range(episode_length):
            state = get_state(ttn)

            thisQ = restrict_move_reward_dict(Qdict, ttn)
            # print('restricted Q dict')
            move = eps_greedy(thisQ, epsilon)

            # print('chose move')

            execute_move(ttn, move, max_bond)
            # print('executed move')

            r = get_reward(ttn, mpo, L, max_bond)
            if r > best_Rs[-1]:
                best_Rs.append(r)
                improved_steps.append(j * episode_length + k)
                best_state = ttn.copy()
            # print('DMRG done')

            Qdict = extend_move_reward_dict(Qdict, ttn)
            # print('extended Qdict')

            max_next = look_ahead(Qdict, ttn)

            Qdict[(state, move)] = ((1 - learning_rate) * Qdict[(state, move)]
                                    + learning_rate * (r + discount * max_next))


            # print('updated Qdict')
            print(f'\rdone {k}', end='')

    return Qdict, best_Rs, best_state, improved_steps
