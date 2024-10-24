#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
dependent_rounding.py

Author: Thomas Wrona <tomdwrona@gmail.com>

Description: A file that performs rounding using cpp (via ctypes) or python.
Dependent, stochastic, or standard rounding. Centers around the round_matrix function.
"""

import os
import time
import random
from itertools import product

import numpy as np
from ctypes import *

# We need to do ctype setup for the CPP functions only once, so do it on import
# Get file location of and load shared object
dependent_rounding_so = CDLL(os.path.join(os.path.dirname(__file__), 'dependent.so'))
# This is what is used to passed 2d arrays into the cpp functions (as reference,
# so values are edited in cpp instead of passing values between)
ND_POINTER = np.ctypeslib.ndpointer(dtype = np.float64, ndim = 2, flags =("C","W"))

#dependent rounding takes array, dimension 1 size, dimension 2 size, random seed, logging
dependent_rounding_so.dependent_rounding.argtypes = ND_POINTER, c_size_t, c_size_t, c_size_t, c_size_t
dependent_rounding_so.dependent_rounding.restype = None

#stochastic rounding takes array, dimension 1 size, dimension 2 size, random seed
dependent_rounding_so.stochastic_rounding.argtypes = ND_POINTER, c_size_t, c_size_t, c_size_t
dependent_rounding_so.stochastic_rounding.restype = None

#standard rounding takes array, dimension 1 size, dimension 2 size
dependent_rounding_so.standard_rounding.argtypes = ND_POINTER, c_size_t, c_size_t
dependent_rounding_so.standard_rounding.restype = None

def dependent_rounding_cpp(array, log = 0, seed = None):
    """Uses shared object to call CPP dependent rounding function on an array.

    Args:
        array (2D numpy float array): Array to be rounded.
        log (int, optional): Whether to print more detailed timing info. Defaults to 0.
        seed (int, optional): Random seed. Defaults to None.

    Returns:
        2D numpy float array: rounded array.
    """
    if not seed: #grab a seed from datetime if none given
        seed = int((time.time() % 1.0)*1000000)
    #call shared object function which was setup on import
    dependent_rounding_so.dependent_rounding(array, array.shape[0], array.shape[1], seed, log)
    return array

def stochastic_rounding_cpp(array, seed = None):
    """Uses shared object to call CPP stochastic rounding function on an array.

    Args:
        array (2D numpy float array): Array to be rounded.
        seed (int, optional): Random seed. Defaults to None.

    Returns:
        2D numpy float array: rounded array.
    """
    if not seed: #grab a seed from datetime if none given
        seed = int((time.time() % 1.0)*1000000)
    #call shared object function which was setup on import
    dependent_rounding_so.stochastic_rounding(array, array.shape[0], array.shape[1], seed)
    return array

def standard_rounding_cpp(array):
    """Uses shared object to call CPP standard rounding function on an array.

    Args:
        array (2D numpy float array): Array to be rounded.
    
    Returns:
        2D numpy float array: rounded array.
    """
    #call shared object function which was setup on import
    dependent_rounding_so.standard_rounding(array, array.shape[0], array.shape[1])
    return array

def round_butterfly(array, edges, left_index_1, left_index_2, right_index_1, right_index_2):
    """Performs 1 dependent runding step on a butterfly (4-cycle) within a given array.
    One of the 4 edges will be destroyed (set to 0/1) by this operation.

    Args:
        array (2D numpy float array): Array to be rounded.
        edges (2D numpy bool array): Array representing whether each edge (array entry) still exists.
        left_index_1 (int): First vertex on left that forms part of butterfly.
        left_index_2 (int): Second vertex on left that forms part of butterfly.
        right_index_1 (int): First vertex on right that forms part of butterfly.
        right_index_2 (int): Second vertex on right that forms part of butterfly.

    Returns:
        int: Vertex on right that has had both edges to it from the left remaining.
    """
    # alist is alternating distance to 0/1 of the cycle
    alist = [1 - array[left_index_1][right_index_1], array[left_index_2][right_index_1],
             1 - array[left_index_2][right_index_2], array[left_index_1][right_index_2]]
    # a is highest possible value to round in one direction
    a = min(alist)
    a_index = alist.index(a)
    # b is highest possible value to round in other direction
    b = max(alist)
    b_index = alist.index(b)
    b = 1 - b
    # randomly choose whether rounding in a/b direction based on proportion
    # retains unbiasedness
    if random.random() < b/(a+b):
        ab = a
        ab_index = a_index
    else:
        ab = -b
        ab_index = b_index
    # alternating add/subtract or subtract/add of resulting value
    array[left_index_1][right_index_1] += ab
    array[left_index_1][right_index_2] -= ab
    array[left_index_2][right_index_2] += ab
    array[left_index_2][right_index_1] -= ab

    # pick out the edge that has been destroyed (set to 0/1) by the above
    if ab_index == 0:
        # remove precision errors (near 0/1 changed to exactly 0/1)
        array[left_index_1][right_index_1] = np.round(array[left_index_1][right_index_1])
        # kill edge in edges for easier calculation later
        edges[left_index_1][right_index_1] = False
        # return other right index
        return right_index_2
    elif ab_index == 1:
        array[left_index_2][right_index_1] = np.round(array[left_index_2][right_index_1])
        edges[left_index_2][right_index_1] = False
        return right_index_2
    elif ab_index == 2:
        array[left_index_2][right_index_2] = np.round(array[left_index_2][right_index_2])
        edges[left_index_2][right_index_2] = False
        return right_index_1
    else:
        array[left_index_1][right_index_2] = np.round(array[left_index_1][right_index_2])
        edges[left_index_1][right_index_2] = False
        return right_index_1

def round_all_butterflies(array, edges):
    """Go through and check all possible butterfly (4-cycle) combinations as
    quickly as possible, calling a dependent rounding step any time any is found.
    When done, no butterflies will remain and most edges will be rounded.

    Args:
        array (2D numpy float array): Array to be rounded.
        edges (2D numpy bool array): Array representing whether each edge (array entry) still exists.
    """
    m = array.shape[0]
    n = array.shape[1]

    #u1, u2 are left vertices, v1, v2 are right vertices
    for u1 in range(m):
        for v1 in range(n):
            # shortcut as much as possible by checking whether edges exist early
            if edges[u1][v1]:
                for u2 in range(u1+1, m):
                    # need to recheck u1 v1 edge in case it was removed in a below step
                    if edges[u1][v1] and edges[u2][v1]:
                        for v2 in range(v1+1, n):
                            # if all edges exist, we round the butterfly
                            # if v1 has both remaining edges, can continue to search
                            # for a new v2. Otherwise, will need to go back out to find
                            # new v1
                            if edges[u1][v2] and edges[u2][v2] and round_butterfly(array, edges, u1, u2, v1, v2) == v2:
                                break 

def adjacency_matrix_from_edges(edges):
    """Create adjacency matrix for a graph.

    Args:
        edges (2D numpy bool array): Array representing whether each edge (array entry) still exists.

    Returns:
        array of sets: adjacency matrix (set of adjacent vertices for each vertex)
    """
    m = edges.shape[0]
    n = edges.shape[1]
    # create a set for adjacency for all vertices.
    adj_matrix = [set() for _ in range(m + n)]
    for c,d in product(range(m),range(n)):
        if edges[c][d]:
            # need to add m to vertices on the right to create new index
            adj_matrix[c].add(m+d)
            adj_matrix[m+d].add(c)
    return adj_matrix

def get_forest(adj_matrix, m, n):
    """Create forest (via array of parents) from adjacency matrix

    Args:
        adj_matrix (array of sets): adjacency matrix (set of adjacent vertices for each vertex)
        m (int): size of dimension 1 of original array
        n (int): size of dimension 2 of original array

    Returns:
        array of int: parent array where each element is the vertex at that index's parent.
    """
    # parents is pointer array (i.e. each element is the parent of that vertex)
    parents = [-1 for _ in range(m+n)]
    to_visit = []

    for i in range(m+n):
        to_visit.append(i)
        while(to_visit):
            curr = to_visit.pop()
            # if no current parent, this vertex is a root
            if parents[curr] == -1:
                parents[curr] = curr
            # each adjacent vertex is a child of the current vertex if not already
            # claimed, and then is visited next
            for n in adj_matrix[curr]:
                if (parents[n] == -1 or parents[n] == n) and parents[curr] != n:
                    parents[n] = curr
                    to_visit.append(n)
    return parents

def tree_edges_from_adj_matrix(adj_matrix, m, n):
    """Takes an adjacency array and generates tree edges for a spanning forest.

    Args:
        adj_matrix (array of sets): adjacency matrix (set of adjacent vertices for each vertex)
        m (int): size of first dimension of original array
        n (int): size of second dimension of original array

    Returns:
        array of sets: sorted list of tree edge vertices adjacent to each vertex
    """
    # get a spanning forest
    forest = get_forest(adj_matrix, m, n)
    tree_edges_unsorted = [[] for _ in range(m+n)]
    for i in range(m+n):
        if forest[i] != i:
            # if an edge in the forest, add each vertex to the other's set
            tree_edges_unsorted[i].append(forest[i])
            tree_edges_unsorted[forest[i]].append(i)
    return [sorted(x) for x in tree_edges_unsorted]

def euler_tour_from_tree(tree_edges, next):
    """Uses tree edges to generate an Euler tour, which is a cycle
    https://en.wikipedia.org/wiki/Euler_tour_technique

    Args:
        tree_edges (array of sets): sorted list of tree edge vertices adjacent to each vertex
        next (2D numpy int array): for tree edge u, v, next[u][v] = w means the next tree edge
                                   in the tour will be v, w

    Returns:
        array of int: Euler tour cycle, in the form of the order of vertex visitation
    """
    # start from first tree edge out of first vertex
    curr1, curr2 = 0, tree_edges[0][0]
    # save copies to find end of loop
    first, second = curr1, curr2
    cycle = [curr1, curr2]
    curr1, curr2 = curr2, next[curr1][curr2]
    # until we are back at the first tree edge, add in next vertex and cycle values
    while(curr1 != first or curr2 != second):
        cycle.append(curr2)
        # use next matrix to continually find next tree edge
        curr1, curr2 = curr2, next[curr1][curr2]
    return cycle

def path_from_euler_tour(tour, u, v):
    """Finds a path between u and v in the given tour, where u and v only occur
    at the start/end of the path

    Args:
        tour (array of int): Euler tour cycle, in the form of the order of vertex visitation
        u (int): element 1 to find
        v (int): element 2 to find

    Returns:
        array of int: path from u to v, subarray of the tour
    """
    maxu = -1
    maxv = -1
    # go from end of tour to find last occurrence of u and v 
    for i in range(len(tour)-1, -1, -1):
        if tour[i] == u and maxu == -1:
            maxu = i
        if tour[i] == v and maxv == -1:
            maxv = i
        if maxu != -1 and maxv != -1:
            break
    # path will be from the last occurrence of one of u, v to the next occurrence of the other
    # i.e. lower of the last occurrences to next of other
    # therefore we have a path between u and v with neither element being present in
    # the interior of the path
    if maxv > maxu:
        # find next place a v occurs
        minv = 0
        for j in range(maxu+1, len(tour)):
            if tour[j] == v:
                minv = j
                break
        # return path, and start/end indices in cycle
        return tour[maxu:minv+1]+[u], maxu, minv
    else:
        #find next place a u occurs
        minu = 0
        for j in range(maxv+1, len(tour)):
            if tour[j] == u:
                minu = j
                break
        # return path, and start/end indices in cycle
        return tour[maxv:minu+1]+[v], maxv, minu

def shorten_visit(visit):
    """Takes a path in an Euler tour and shortens it to minimal path by removing
    any edges that have the reverse edge also in the path

    Args:
        visit (array of int): path to be shortened

    Returns:
        array of int: shortened path
    """
    edges = set()
    # add each edge into a set
    for i in range(0,len(visit)-1):
        edges.add((visit[i],visit[i+1]))
    # edges remain in the path if their reverse edge is not in the set 
    # so if both directed edges between two vertices exist, both edges are removed
    newvisit = []
    for i in range(0,len(visit)-1):
        if ((visit[i+1],visit[i]) not in edges):
            newvisit.append(visit[i])
    newvisit.append(visit[-1])

    return newvisit

def recreate_cycle_without_edge(tour, next, visit, i, buffer, bufferend):
    """Given an Euler tour, an edge to delete, and an edge to replace it with, we
    want to remove both directions of the deleted edge and recreate an Euler tour by
    putting the replacement edge (also both directions) in. This is possible because
    removing one edge from a spanning tree and adding a new edge in retains the spanning tree.
    
    We remove deleted edges by taking ... u,v ... v,u ... and finding a way to knit the
    part before the first u and after the second u together, as well as a part before
    the second v and after the first v together.

    Adding a new edge is roughly the opposite operation.

    Args:
        tour (array of int): Euler tour cycle, in the form of the order of vertex visitation
        visit (array of int): shortened path from u to v
        i (int): index of deleted edge in visit
        buffer (int): start of original path from u to v (i.e. location of u or v)
        bufferend (int): end of original path from u to v (i.e. location of u or v)

    Returns:
        tour (array of int): updated Euler tour cycle, in the form of the order of vertex visitation
    """
    # get locations of the ith edge from visit in the tour
    reverse_loc = 0
    for reverse_loc in range(1,len(tour)):
        if tour[reverse_loc-1] == visit[i] and tour[reverse_loc] == visit[i-1]:
            break
    forward_loc = 0
    for forward_loc in range(1,len(tour)):
        if tour[forward_loc] == visit[i] and tour[forward_loc-1] == visit[i-1]:
            break
    #we do nothing if forward location is right at the end (would be removing and adding same edge)
    if(forward_loc < len(tour)-1):
        # need to update euler tour. the rounded edge visit[i-1],visit[i] is removed,
        # the u, v edge is added
        # delete visit[i-1],visit[i] and visit[j-1], visit[j]
        # add in u,v --  i-2  i-1 i  i+1     j-2 j-1 j j+1   to
        #                 i-2 i-1 j+1           j-2 j-1 i+1
        if reverse_loc < forward_loc:
            tour = tour[:reverse_loc]+tour[forward_loc+1:bufferend+1]+tour[buffer:forward_loc]+tour[reverse_loc+1:buffer+1]+tour[bufferend:]
        else:
            tour = tour[:buffer+1]+tour[bufferend:reverse_loc]+tour[forward_loc+1:bufferend+1] + tour[buffer:forward_loc]+tour[reverse_loc+1:]
        next[visit[i]][visit[i-1]] = next[visit[i-1]][visit[i]] = -1 #remove lr edge from loop
        next[visit[0]][visit[-2]] = next[visit[-2]][visit[0]] = 1 #indicate this is now tree edge
    return tour

def round_path(array, next, visit, tour, buffer, bufferend):
    """Take a path and round the edges on it, then replace the rounded edge with
    new edges in the tour.

    Args:
        array (2D numpy float array): Array to be rounded.
        next (2D numpy int array): Succession array for tree.
        visit (array of int): minimal path between vertices in replacement edge
        tour (array of int): Euler tour cycle, in the form of the order of vertex visitation
        buffer (int): start index of visit in tour
        bufferend (int): end index of visit in tour

    Returns:
        array of int: new Euler tour
    """
    near_zero = 1e-10
    m = array.shape[0]
    a, b = 1.0, 1.0
    # first remember that a dependent rounding step will alternate adding and subtracting edges
    # a will be the max amount that the first element can be rounded up, second element down, etc.
    # so a is min of 1 - even elements and odd elements
    # b is opposite, so min of even elements and 1 - odd elements
    for i in range(1,len(visit)):
        v1, v2 = (visit[i], visit[i-1] - m) if visit[i-1] >= m else (visit[i-1], visit[i] - m)
        a, b = (min(a, 1-array[v1][v2]), min(b, array[v1][v2])) if (i % 2 == 0) else (min(a, array[v1][v2]), min(b, 1-array[v1][v2]))

    # randomly select whether to round up or down based on proportion
    ab = 0 if (a == 0 or b == 0) else (a if random.random() < b/(a+b) else -b)
    # round path
    for i in range(1,len(visit)):
        v1,v2 = (visit[i], visit[i-1] - m) if visit[i-1] >= m else (visit[i-1], visit[i] - m)
        array[v1][v2] += (-1)**(i)*ab
        # if this is the element that has been rounded (i.e. close enough to 0/1) 
        # then it will be removed from the cycle
        if array[v1][v2] < near_zero or array[v1][v2] > 1-near_zero:
            array[v1][v2] = round(array[v1][v2])
            # remove rounded edge from cycle, add in replacement edge
            tour = recreate_cycle_without_edge(tour, next, visit, i, buffer, bufferend)
    return tour

def round_cycles(array, edges):
    """Rounds all cycles formed by edges that still remain to be rounded, meaning
    the edges that remain to be rounded form a tree.

    Args:
        array (2D numpy float array): Array to be rounded.
        edges (2D numpy bool array): Array representing whether each edge (array entry) still exists.

    Returns:
        array of ints: Euler tour of remaining tree
    """
    m = array.shape[0]
    n = array.shape[1]

    adj_matrix = adjacency_matrix_from_edges(edges)
    
    # get ordered tree edges for a spanning tree
    tree_edges = tree_edges_from_adj_matrix(adj_matrix, m, n)
    # next matrix is setup so that for a tree edge u, v, next[u][v] = w means
    # that v, w is also a tree edge which we treat as the successor.
    next = np.full((m+n,m+n),-1,dtype=int)
    for v in range(m+n):
        for i in range(len(tree_edges[v])):
            # we generate the next matrix based on the tree edge ordering
            # for vertex v's ith adjacent vertex, we set the next edge of the
            # edge in from that ith vertex to v as the edge from v to vertex i+1
            next[tree_edges[v][i]][v] = tree_edges[v][(i+1) % len(tree_edges[v])]
    # setup for cycle rounding by getting an Euler tour.
    tour = euler_tour_from_tree(tree_edges, next)

    # now we find all non-tree edges and round cycles containing them away. 
    # But because we use the Euler tour to find paths, we maintain the Euler tour
    # by adding the non-tree edge into the Euler tour (if it was not the rounded edge)
    for u in range(m):
        for v in adj_matrix[u]:
            # if not in tree
            if next[u][v] == -1: 
                # get path from u to v
                visit, buffer, bufferend = path_from_euler_tour(tour, u, v)
                # trim path so that it is minimal (problems arise if we have 
                # both directed edges corresponding to the same edge when rounding)
                visit = shorten_visit(visit)
                # round path and replace the rounded edge with u,v
                tour = round_path(array, next, visit, tour, buffer, bufferend)
         
    return tour

def get_leafs(tour):
    """Gets first two leafs in Euler tour of tree.

    Args:
        tour (array of int): Euler tour of tree

    Returns:
        int, int: first two leaf vertex indices
    """
    # leaf is where vertex before and after are the same
    firsti = 1
    while tour[firsti-1] != tour[firsti+1]:
        firsti += 1
        
    #get second leaf closest after first
    secondi = firsti + 1
    while secondi < len(tour)-1 and tour[secondi-1] != tour[(secondi+1) % len(tour)]:
        secondi += 1

    return firsti, secondi

def round_tree(array, tour):
    """Rounds all remaining edges, which form a tree due to all cycles being removed.
    Finds minimal paths in the tree (between two leafs), rounds those paths,
    then repeats the process on the two split components.

    Args:
        array (2D numpy float array): Array to be rounded.
        tour (array of int): Euler tour of tree
    """
    if len(tour) > 1:
        near_zero = 1e-10
        m = array.shape[0]

        # get first two leafs so we do not need to shorten path between them for rounding
        first_leaf_index, second_leaf_index = get_leafs(tour)

        # similar to round_path, but instead of recreating cycle, it just removes edge
        # and calls round_tree on the two split Euler tours.
        # a is min of 1 - even elements and odd elements
        # b is opposite, so min of even elements and 1 - odd elements
        a, b = 1.0, 1.0
        for i in range(first_leaf_index+1, second_leaf_index+1):
            v1, v2 = (tour[i], tour[i-1] - m) if tour[i-1] >= m else (tour[i-1], tour[i] - m)
            a, b = (min(a, 1 - array[v1][v2]), min(b, array[v1][v2])) if (i % 2 == 0) else (min(a, array[v1][v2]), min(b, 1 - array[v1][v2]))

        # randomly select whether to round up or down based on proportion
        ab = 0 if (a == 0 or b == 0) else (a if random.random() < b/(a+b) else -b)
        # round path
        backwards_loc, forward_loc = 0, 0
        for i in range(first_leaf_index+1, second_leaf_index+1):
            v1, v2 = (tour[i], tour[i-1] - m) if tour[i-1] >= m else (tour[i-1], tour[i] - m)
            array[v1][v2] += (-1)**i * ab
            # if this is the element that has been rounded (i.e. close enough to 0/1) 
            if array[v1][v2] < near_zero or array[v1][v2] > 1 - near_zero:
                array[v1][v2] = round(array[v1][v2])
                forward_loc = i
                # get location of reverse edge for rounded edge
                for backwards_loc in range(1, len(tour)):
                    if tour[backwards_loc-1] == tour[i] and tour[backwards_loc] == tour[i-1]:
                        break
        # split tree at rounded edge and repeat process on remaining halves
        if backwards_loc < forward_loc:
            round_tree(array, tour[backwards_loc:forward_loc])
            round_tree(array, tour[forward_loc:-1]+tour[:backwards_loc])
        else:
            round_tree(array, tour[forward_loc:backwards_loc])
            round_tree(array, tour[backwards_loc:-1]+tour[:forward_loc])

def dependent_rounding_py(array, log, seed):
    """Dependent rounding on given array. For details on the original algorithm
    and theoretical properties, see
    https://www.cs.umd.edu/~samir/grant/jacm06.pdf

    Args:
        array (2D numpy float array): Array to be rounded.
        log (int): whether to log details about rounding per stage.
        seed (int): random seed.

    Returns:
        2D numpy float array: rounded array
    """
    if not seed: #grab a seed from datetime if none given
        seed = int((time.time() % 1.0)*1000000)
    random.seed(seed)

    if log:
        start_time = time.time()
        stage_start_time = start_time

    # get edge matrix
    edges = np.ones_like(array, dtype = bool)
    round_all_butterflies(array, edges)

    if log:
        stage_time_elapsed = (time.time() - stage_start_time)*1000.0
        num_edges_remaining = np.sum((array != np.floor(array)))
        print("Butterfly rounding completed")
        print("Edges rounded: "+str(array.size - num_edges_remaining))
        print("Time elapsed (ms): "+str(stage_time_elapsed))
        print("Avg time per 1000 rounded elements (ms): "+str(stage_time_elapsed * 1000.0 / max(1,array.size - num_edges_remaining)))
        print()
        stage_start_time = time.time()
        last_stage_remaining = num_edges_remaining

    # will get Euler tour to feed into round_tree
    tour = round_cycles(array, edges)

    if log:
        stage_time_elapsed = (time.time() - stage_start_time)*1000.0
        num_edges_remaining = np.sum((array != np.floor(array)))
        print("Cycle rounding completed")
        print("Edges rounded: "+str(last_stage_remaining - num_edges_remaining))
        print("Time elapsed (ms): "+str(stage_time_elapsed))
        print("Avg time per 1000 rounded elements (ms): "+str(stage_time_elapsed * 1000.0 / max(1,last_stage_remaining - num_edges_remaining)))
        print()
        stage_start_time = time.time()
        last_stage_remaining = num_edges_remaining

    round_tree(array, tour)

    if log:
        stage_time_elapsed = (time.time() - stage_start_time)*1000.0
        total_time_elapsed = (time.time() - start_time)*1000.0
        print("Tree rounding completed")
        print("Edges rounded: "+str(last_stage_remaining))
        print("Time elapsed (ms): "+str(stage_time_elapsed))
        print("Avg time per 1000 rounded elements (ms): "+str(stage_time_elapsed * 1000.0 / max(1,last_stage_remaining)))
        print()
        print("Total time elapsed (ms): "+str(total_time_elapsed))
        print("Overall avg time per 1000 rounded elements (ms): "+str(total_time_elapsed * 1000.0 / array.size))

    return array

def stochastic_rounding_py(array, seed):
    """Stochastic rounding takes an array and independently rounds each element
    as follows: if element is below a uniform random number between 0 and 1,
    it is rounded to 1. Otherwise it is rounded to 0. The result is unbiased.

    Args:
        array (2D numpy float array): Array to be rounded.
        seed (int): random seed.

    Returns:
        2D numpy float array: rounded array.
    """
    if not seed: #grab a seed from datetime if none given
        int((time.time() % 1.0)*1000000)
    np.random.seed(seed)
    # if random number is below element, set to true/1.0, otherwise set to false/0.0
    return (np.random.rand(*array.shape) < array).astype(np.float64)

def standard_rounding_py(array):
    """Standard rounding rounds each element independently (set to closest int)

    Args:
        array (2D numpy float array): Array to be rounded.

    Returns:
        2D numpy float array: rounded array.
    """
    return np.round(array)

def round_matrix(array, method = "dependent", resolution = 0, language = "cpp", seed = None, log = 0, track_time = False):
    """Catchall round function. Easy to use for comparison of different languages and methods of rounding.

    Args:
        array (numpy float array): array to round. Sometimes does inplace changes.
        method (string, optional): "dependent", "stochastic", or "standard".
        resolution (int or "sign", optional): if "sign", rounding to +/-1. Otherwise, rounds 
                                              to nearest 2^resolution. Defaults to 0.
        language (string, optional): "cpp" or "python". Defaults to "cpp".
        seed (int, optional): Random seed (based on current time if none given). Defaults to None.
        log (int, optional): Whether dependent rounding functions will put extra logging on 
                             rounding stages. Defaults to 0.
        track_time (bool, optional): Whether to print out total ms taken to round. Defaults to False.

    Returns:
        numpy float array: rounded array.
    """
    if track_time:
        start_time = time.time()

    # need to perform some operations on the array to put all values between 0 and 1
    if resolution == "sign":
        array_to_round = (np.minimum(np.maximum(array,-1.0),1.0)+1.0)/2
    else:
        array_to_round = array / 2.0**resolution
        base_matrix = np.floor(array_to_round)
        array_to_round -= base_matrix
    # array can be any size (greater than 2 dimensions). Dimensions above 2 are all
    # flattened into one.
    orig_dims = array_to_round.shape
    if len(orig_dims) == 1:
        array_to_round = array_to_round.reshape((-1, orig_dims[-1], 1), order = 'C')
    else:
        array_to_round = array_to_round.reshape((-1, orig_dims[-2], orig_dims[-1]), order = 'C')
    # It is faster if the first dimension being rounded is smaller
    #if orig_dims[-1] < orig_dims[-2]:
    #    array_to_round = np.swapaxes(array_to_round, -2, -1)
    # pick rounding function based on args
    if language == "cpp":
        if method == "dependent":
            round_function = lambda layer: dependent_rounding_cpp(layer, log, seed)
        elif method == "stochastic":
            round_function = lambda layer: stochastic_rounding_cpp(layer, seed)
        else:
            round_function = lambda layer: standard_rounding_cpp(layer)
    else:
        if method == "dependent":
            round_function = lambda layer: dependent_rounding_py(layer, log, seed)
        elif method == "stochastic":
            round_function = lambda layer: stochastic_rounding_py(layer, seed)
        else:
            round_function = lambda layer: standard_rounding_py(layer)

    # rounding step
    for two_dim_slice_index in range(array_to_round.shape[0]):
        array_to_round[two_dim_slice_index] = round_function(array_to_round[two_dim_slice_index])

    # if transposed, swap back
    #if orig_dims[-1] < orig_dims[-2]:
    #    array_to_round = np.swapaxes(array_to_round, -2, -1)
    array_to_round = np.reshape(array_to_round, orig_dims)

    # return values to original ranges
    if resolution == "sign":
        array_to_round = array_to_round * 2.0 - 1.0
    else:
        array_to_round = (array_to_round + base_matrix) * 2.0**resolution

    if track_time:
        print("Total time (ms): " + str((time.time()-start_time)* 1000.0))
    
    return array_to_round