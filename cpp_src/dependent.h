/**
 * @file
 * @author  Thomas Wrona <tomdwrona@gmail.com>
 *
 * @section DESCRIPTION
 *
 * This file contains a dependent rounding function and associated utilities.
 */

#ifndef DEPENDENT_ROUND_H
#define DEPENDENT_ROUND_H

#include <parlay/primitives.h>

// Minor use, tuple of two size_t vertex indices sometimes represents a bipartite graph edge
using BipartiteEdge = std::tuple<size_t,size_t>;
// A path is a sequence of vertex indices
using Path = parlay::sequence<size_t>;
// 2D matrix of booleans to represent whether a given edge exists
using EdgeIndicators = parlay::sequence<parlay::sequence<bool>>;
// 2D matrix of doubles to represent the current value of all edges
using DoubleMatrix = parlay::sequence<parlay::sequence<double>>;
// 2D matrix where the first index represents a vertex
// each vertex has an associated adjacency array
using AdjacencyMatrix = parlay::sequence<parlay::sequence<size_t>>;
// 2D matrix where the first and second indexes are the two vertices in an edge
// and the contained value will be the next or previous element in a path.
// Optional for if no next value exists.
using SuccessionMatrix = parlay::sequence<parlay::sequence<std::optional<size_t>>>;

/**
* Function that takes a path (list of vertices) and produces a list of bipartite edges
* that form that path.
* 
* m is needed because the path indexes vertices of both
* partitions together and the bipartite edges index them separately (i.e. it needs
* to subtract m from the one that is above m).
*
* @param path is a reference to a path (list of vertices).
*
* @param m is the first dimension size of the input array.
*
* @return a list of bipartite edges that form the given path.
*/
parlay::sequence<BipartiteEdge> bipartite_edges_from_path(Path &path, size_t m);

/**
* Function that takes a path (list of vertices) and performs one rounding iteration
* on it (i.e. ends with one value rounded to 0 or 1).
* 
* The path should be a cycle or maximal path. The odd- and even- indexed edges in
* the path will have their values in the array increased and decreased, respectively,
* in order to get one edge to either 0 or 1.
*
* @param array is a reference to the original array.
*
* @param path is a reference to the path to be rounded.
*
* @return The index in the path of the rounded edge.
*/
size_t round_path(DoubleMatrix &array, Path &path);

/**
* Function that takes a butterfly (i.e. a 4-cycle, or two vertices in each partition such that
* all four possible edges exist) and performs one rounding iteration
* on it (i.e. ends with one value rounded to 0 or 1).
*
* @param array is a reference to the original array.
*
* @param edges is a reference to an array that indicates whether each edge still
*        exists or not.
*
* @param left_element_1 is the index in the left partition of one vertex.
* 
* @param left_element_2 is the index in the left partition of a second vertex.
*
* @param right_element_1 is the index in the right partition of one vertex.
* 
* @param right_element_2 is the index in the right partition of a second vertex.
*
* @return The index in the right partition where both edges to it from the left
*         still exist.
*/
size_t round_butterfly(DoubleMatrix &array, EdgeIndicators &edges,
                       size_t left_element_1, size_t left_element_2,
                       size_t right_element_1, size_t right_element_2);


/**
* In this namespace is an alternative method to round all butterflies.
* It splits the left partition into pairs and gives each pair to a processor
* for concurrent rounding.
* It was found to be slower in practice, but it is possible that it outperforms
* the current method with an increase in processor number.
*/
namespace split_left{
    /**
    * Function that rounds all butterflies by iterating through all possibilities.
    *
    * @param array is a reference to the original array.
    *
    * @param edges is a reference to an array that indicates whether each edge still
    *        exists or not.
    */
    void round_all_butterflies(DoubleMatrix &array, EdgeIndicators &edges);
}

/**
* In this namespace is an alternative method to round all butterflies.
* It splits the left partition into pairs and gives each pair to a processor
* for concurrent rounding. It will then locate all wedges from that pair
* and use further multithreading to round those in parallel as much as possible.
* It was found to be slower in practice, but it is possible that it outperforms
* the current method with an increase in processor number.
*/
namespace split_leftright {
    /**
    * Function that rounds all butterflies by iterating through all possibilities.
    *
    * @param array is a reference to the original array.
    *
    * @param edges is a reference to an array that indicates whether each edge still
    *        exists or not.
    */
    void round_all_butterflies(DoubleMatrix &array, EdgeIndicators &edges);
}

/**
* Function that rounds all butterflies by iterating through all possibilities.
*
* @param array is a reference to the original array.
*
* @param edges is a reference to an array that indicates whether each edge still
*        exists or not.
*/
void round_all_butterflies(DoubleMatrix &array, EdgeIndicators &edges);

/**
* Function that "trims" a path on a tree by removing leaves until
* only the shortest path remains.
*
* @param path is a reference to the path to trim.
*/
void trim_path(Path &path);

/**
* Function that takes an edge indicator array and produces the corresponding
* adjacency matrix.
*
* @param edges is a reference to an array that indicates whether each edge still
*        exists or not.
*
* @return The adjacency matrix.
*/
AdjacencyMatrix adjacency_matrix_from_edges(EdgeIndicators &edges);

/**
* Function that takes an adjacency matrix for a graph, produces a spanning tree
* for that graph, and returns the adjacency matrix for the spanning tree.
*
* @param adj_matrix is a reference to the adjacency matrix.
*
* @return The adjacency matrix for a spanning tree.
*/
AdjacencyMatrix tree_edges_from_adj_matrix(AdjacencyMatrix &adj_matrix);

/**
* Function that takes an adjacency matrix for tree edges and a succession matrix
* that contains the next vertex along the path per edge to create an Euler Tour.
* 
* This function does little other than take the matrix and run through it to
* produce an ordering, which becomes an Euler tour.
* 
* See https://en.wikipedia.org/wiki/Euler_tour_technique.
*
* @param tree_edges is a reference to the adjacency matrix for tree edges.
*
* @param next_matrix is a reference to a succession matrix that indicates the next
*        vertex along the path for each edge.
*
* @return The Euler tour corresponding to the next matrix.
*/
Path euler_tour_from_tree(AdjacencyMatrix &tree_edges, SuccessionMatrix &next_matrix);

/**
* Function that takes an Euler tour and two elements to find a path between them
* and return it and the start/end indices.
*
* @param tour is the Euler tour.
* 
* @param left_element is one element to search for.
* 
* @param right_element is the second element to search for.
* 
* @param start_index is a reference to put the start index of the resulting path in.
* 
* @param end_index is a reference to put the end index of the resulting path in.
*
* @return A path starting from one given element and ending at another.
*/
Path path_from_euler_tour(Path &tour, size_t left_element, size_t right_element,
                          size_t &start_index, size_t &end_index);

/**
* Function that pushes a segment of a path between two indices into a container.
*
* @param new_path_segments is a reference to the container of paths which will
*        later be flattened.
* 
* @param path is the path to take from.
* 
* @param start_index is the start index to take from.
* 
* @param end_index is the end index to take from.
*/
void push_path_segment(parlay::sequence<parlay::sequence<size_t>> &new_path_segments,
                       Path path, size_t start_index, size_t end_index);

/**
* Function that takes a cycle and recreates it after removing an edge.
*
* @param cycle is a reference to the cycle to recreate.
* 
* @param next_matrix is a reference to the succession matrix of the cycle.
* 
* @param visited_path is a reference to the path which the edge is in.
* 
* @param edge_index is the index of the edge to remove in the visited path.
* 
* @param path_start is where visited_path starts in the cycle.
* 
* @param path_end is where visited_path ends in the cycle.
*/
void recreate_cycle_without_edge(Path &cycle, SuccessionMatrix &next_matrix,
                                 Path &visited_path, size_t edge_index,
                                 size_t path_start, size_t path_end);

/**
* Function that rounds all cycles remaining in the graph corresponding to an array.
*
* @param array is a reference to the original array.
*
* @param edges is a reference to an array that indicates whether each edge still
*        exists or not.
* 
* @return an Euler tour of a spanning tree of the graph corresponding to the array,
*         after all cycles have been removed.
*/
Path round_all_cycles(DoubleMatrix &array, EdgeIndicators &edges);

/**
* Function that rounds all the spanning tree of edges remaining after all cycles
* were removed.
*
* @param array is a reference to the original array.
* 
* @param tour is a reference to the Euler tour of the spanning tree.
*/
void round_tree(DoubleMatrix &array, Path &tour);

/**
* Function that performs dependent rounding on an array. For details on the original algorithm
* and theoretical properties, see
* https://www.cs.umd.edu/~samir/grant/jacm06.pdf
*
* @param array is the input array to round size mn (flattened 2d m x n array).
*        It is rounded in place.
*
* @param m is the first dimension size.
* 
* @param n is the second dimension size.
* 
* @param seed is the random seed used.
*
* @param logging_level is how much to print to cout. 0 means no logging,
         1 means the total time and average time per element will be printed per step.
         Including logging will increase runtime.
*/
extern "C" void dependent_rounding(double *array, size_t m, size_t n, size_t seed, size_t logging_level = 0);

#endif