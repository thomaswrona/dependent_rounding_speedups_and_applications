/**
 * @file
 * @author  Thomas Wrona <tomdwrona@gmail.com>
 *
 * @section DESCRIPTION
 *
 * This file contains a dependent rounding function and associated utilities.
 */

#include <iostream>

#include <parlay/primitives.h>

#include "hash_map.h"
#include "round_util.h"
#include "dependent.h"

parlay::sequence<BipartiteEdge> bipartite_edges_from_path(Path &path, size_t m) {
    // tabulate will take the lambda function and call it on 0, 1,... path.size()-2
    // and return a sequence of the return values.
    return parlay::tabulate(path.size()-1, [&path, &m](size_t i) {
        // extract the vertexes in order
        size_t first_vertex {path[i]};
        size_t second_vertex {path[i+1]};
        // need to determine which one is left partition (index < m)
        // and right partition (index >= m)
        size_t left_vertex {(first_vertex >= m) ? second_vertex : first_vertex};
        size_t right_vertex {first_vertex + second_vertex - left_vertex - m};
        return std::make_tuple(left_vertex, right_vertex);
    });
}

size_t round_path(DoubleMatrix &array, Path &path) {
    // m is first dimension of array
    size_t m {array.size()};

    // start by getting each edge into a better format (list of pairs) instead of path
    parlay::sequence<BipartiteEdge> bipartite_edges {bipartite_edges_from_path(path, m)};
    
    // take each edge and get the "a" vector for the dependent rounding operation
    // we want the min/max alternating along each edge, so we alternate
    // distance to floor and (1 - distance to floor) of the elements.
    parlay::sequence<double> a_vector {parlay::tabulate(path.size()-1, [&bipartite_edges, &array](size_t i) {
        auto [u,v] = bipartite_edges[i];
        // alternating distance to floor and (1 - distance to floor)
        return (1-i%2) - pow(-1,i%2) * distance_to_floor(array[u][v]);
    })};

    // case "a" will be where we raise even index elements and lower odd index elements
    // case "b" will be where we lower even index elements and raise odd index elements
    // so a is minimum of a vector and b is maximum.
    size_t a_index {static_cast<size_t>(parlay::min_element(a_vector) - a_vector.begin())};
    size_t b_index {static_cast<size_t>(parlay::max_element(a_vector) - a_vector.begin())};
    double a {a_vector[a_index]};
    double b {1-a_vector[b_index]};
    
    // use random value to determine which case is used (retain unbiasedness)
    if(static_cast<double> (rand()) / RAND_MAX >= b / (a + b)){
        a = -b;
        a_index = b_index;
    }

    // add/subtract a/b, again this alternates
    parlay::parallel_for(0, path.size()-1, [&](size_t j){
        auto [u, v] = bipartite_edges[j];
        array[u][v] += pow(-1,j)*a;        
    });

    auto [u,v] = bipartite_edges[a_index];
    // round to avoid float errors
    array[u][v] = round(array[u][v]);

    return a_index;
}

size_t round_butterfly(DoubleMatrix &array, EdgeIndicators &edges,
                       size_t left_element_1, size_t left_element_2,
                       size_t right_element_1, size_t right_element_2) {
    // this is a similar method to round_path but unrolled because it is called so 
    // often on butterflies / 4 length cycles.
    // case "a" will be where we raise even index elements and lower odd index elements
    // case "b" will be where we lower even index elements and raise odd index elements
    // b and a (and their indices) are calculated as we go along each index.
    double b {distance_to_floor(array[left_element_1][right_element_1])};
    size_t b_index {0};
    double a {1 - b};
    size_t a_index {0};
    double temp;
    
    //check if second edge on path meets condition for either a or b
    temp = distance_to_floor(array[left_element_2][right_element_1]);
    if (temp < a){
        a = temp;
        a_index = 1;
    }
    if(1 - temp < b) {
        b = 1 - temp;
        b_index = 1;
    }
     //check if third edge on path meets condition for either a or b
    temp = distance_to_floor(array[left_element_2][right_element_2]);
    if (1 - temp < a){
        a = 1 - temp;
        a_index = 2;
    }
    if (temp < b){
        b = temp;
        b_index = 2;
    }
     //check if fourth edge on path meets condition for either a or b
    temp = distance_to_floor(array[left_element_1][right_element_2]);
    if (temp < a){
        a = temp;
        a_index = 3;
    }
    if (1 - temp < b) {
        b = 1 - temp;
        b_index = 3;
    }
    // use random value to determine which case is used (retain unbiasedness)
    if((double)rand() / RAND_MAX >= b/(a+b)){
        a = -b;
        a_index = b_index;
    }
    // add/subtract a/b
    array[left_element_1][right_element_1] += a;
    array[left_element_2][right_element_2] += a;
    array[left_element_2][right_element_1] -= a;
    array[left_element_1][right_element_2] -= a;

    // we kill the edge that has been rounded to 0/1 and return the right vertex that 
    // has both edges to it surviving (i.e. right vertex that is not in destroyed edge)
    size_t left_element{((a_index == 0) || (a_index == 3)) ? left_element_1 : left_element_2};
    size_t right_element{((a_index == 0) || (a_index == 1)) ? right_element_1 : right_element_2};
    edges[left_element][right_element] = false;
    array[left_element][right_element] = round(array[left_element][right_element]);
    return right_element_1 + right_element_2 - right_element;
}

namespace split_left{
    void round_all_butterflies(DoubleMatrix &array, EdgeIndicators &edges){
        // this function is currently slow but may be faster with more processors
        size_t m {array.size()};
        size_t n {array[0].size()};
        // number of rounds for round robin will be m-1 if m is even or m otherwise
        size_t rounds {m - 1 + m % 2};

        for(size_t round {0}; round < rounds; round++){
            // each round we pair off vertices on the left and so can work with 
            // each pair in parallel
            parlay::parallel_for(0, (rounds+1)/2, [&](size_t i) {
                auto [u1,u2] = generate_round_robin(i, round, rounds);
                // if m is odd, then we have been pretending that m is one higher than
                // it actually is, and so if we see u1 or u2 >= m, this pair is not used (a "bye")
                if(u1 < m && u2 < m){
                    std::optional<size_t> v1 {};
                    // iterate through right partition. First vertex that has both
                    // edges to it is stored.
                    // Once a second vertex is found, the butterfly is rounded and
                    // we save the vertex that still has both edges to it and keep looking.
                    for(size_t v2 {0}; v2 < n; v2++)
                        if(edges[u1][v2] && edges[u2][v2])
                            v1 = (v1.has_value())? round_butterfly(array, edges, u1, u2, v1.value(), v2): v2;
                }
            });
        }
    }
}

namespace split_leftright {
    void round_all_butterflies(DoubleMatrix &array, EdgeIndicators &edges){
        // this function is currently slow but may be faster with more processors
        size_t m {array.size()};
        size_t n {array[0].size()};
        // number of rounds for round robin will be m-1 if m is even or m otherwise
        size_t rounds {m - 1 + m % 2};
        
        for(size_t round {0}; round < rounds; round++){
            // each round we pair off vertices on the left and so can work with 
            // each pair in parallel
            parlay::parallel_for(0, (rounds+1)/2, [&](size_t i) {
                auto [u1,u2] = generate_round_robin(i, round, rounds);
                // if m is odd, then we have been pretending that m is one higher than
                // it actually is, and so if we see u1 or u2 >= m, this pair is not used (a "bye")
                if(u1 < m && u2 < m){
                    // in parallel find all wedges (vertices on right partition that
                    // have both edges to our 2 left vertices)
                    auto wedges = parlay::filter(parlay::iota<size_t>(n), [&](size_t j) {
                        return edges[u1][j] && edges[u2][j];
                    });
                    // as long as we have enough wedges to form a butterfly
                    while(wedges.size() > 1){
                        // pair off edges on right and round them in parallel
                        parlay::parallel_for(0,wedges.size()/2, [&](size_t j) {
                            round_butterfly(array, edges, u1, u2, wedges[2*j], wedges[2*j+1]);
                        });

                        // refilter wedges to repeat this process on ones that remain.
                        wedges = parlay::filter(wedges, [&](size_t j) {
                            return edges[u1][j] && edges[u2][j];
                        });
                    }
                }
            });
        }
    }
}

void round_all_butterflies(DoubleMatrix &array, EdgeIndicators &edges){
    // currently best function for rounding butterflies, because it shortcuts
    // so many of the calculations to find butterflies, but it is entirely serial.
    size_t m {array.size()};
    size_t n {array[0].size()};
    // u1 and u2 will be indices in the left partition, v1 and v2 on the right.
    for(size_t u1 {0}; u1 < m; u1++)
        for(size_t v1 {0}; v1 < n; v1++)
            // check if edges exist as early as possible
            if(edges[u1][v1])
                for(size_t u2 {u1+1}; u2 < m; u2++) 
                    // need to check first edge again in case it is destroyed later
                    if(edges[u1][v1] && edges[u2][v1])
                        for(size_t v2 {v1+1}; v2 < n; v2++)
                            // use shortcutting to check both edges then round if all exist
                            // if v2 is the wedge that has survived, then we have to break
                            // out of the loop because we need a new v1
                            // otherwise we can continue in this loop
                            if(edges[u1][v2] && edges[u2][v2] && 
                               round_butterfly(array, edges, u1, u2, v1, v2) == v2)
                                break;
}

void trim_path(Path &path){
    // the essence of this function's implementation is that we need to remove all
    // edges where the reverse edge is also in the path (if u,v and v,u are both
    // part of the path, then both are removed)
    size_t len {path.size()};
    // hash map from parlaylib examples
    hash_map<size_t,bool> reverse_check((long int) len);
    // insert the unique cantor pair of the edge into the hash map.
    parlay::parallel_for(0, len-1, [&](size_t i){
        reverse_check.insert(cantor_pair(path[i], path[i+1]), true);
    });
    // pack will take the elements of path that satisfy the condition given.
    path = parlay::pack(path, parlay::tabulate(len, [&](size_t i) {
        // an edge is removed if the cantor pair of the reverse order is also
        // in the path.
        return (i==0) || !(reverse_check.find(cantor_pair(path[i], path[i-1])));
    }));
}

AdjacencyMatrix adjacency_matrix_from_edges(EdgeIndicators &edges) {
    size_t m {edges.size()};
    size_t n {edges[0].size()};

    AdjacencyMatrix adj_matrix(m+n, parlay::sequence<size_t>(m+n));

    // first note that the edges matrix has both partition vertices indexed starting
    // from zero. To put them both into an adjacency matrix, we start by packing
    // the edges that exist from the edge matrix.
    parlay::parallel_for(0,m, [&](size_t i){
        // pack edges that exist from left to right, then add m to the indices
        // to indicate they are in the right partition for later use
        adj_matrix[i] = parlay::map(parlay::pack_index(edges[i]), [&](size_t x) {return x+m;});
    });
    parlay::parallel_for(m,m+n, [&](size_t i){
        // for edges from right to left, we just need to check the edges matrix while
        // subtracting m from the indices we input for the second dimension
        adj_matrix[i] = parlay::filter(parlay::iota<size_t>(m), [&](size_t x) {return edges[x][i-m];});
    });
    return adj_matrix;
}

AdjacencyMatrix tree_edges_from_adj_matrix(AdjacencyMatrix &adj_matrix) {
    size_t m_plus_n {adj_matrix.size()};

    // DFS based on parlaylib examples
    parlay::sequence<size_t> parents(m_plus_n, SIZE_MAX);
    size_t root {0};
    // create visited array which indicates whether each element was visited
    parlay::sequence<bool> visited = parlay::tabulate(m_plus_n, [&] (size_t i) {
        return (i==root);
    });

    parents[root] = root;
    // frontier is list of vertices to visit
    size_t frontier;
    parlay::sequence<size_t> frontiers(1,root);
    while(frontiers.size() > 0){
        // pick next element
        frontier = frontiers.back();
        frontiers.pop_back();
        // go through adjacency of frontier element and pick out ones that werent visited
        parlay::sequence<size_t> out = parlay::filter(adj_matrix[frontier], [&] (size_t v) {
            if(!visited[v]){
                // if not visited yet, we check it off and set the parent
                visited[v] = true;
                parents[v] = frontier;
                return true;
            } else
                return false;
        });
        // add new frontier
        frontiers = parlay::merge(frontiers, out);
    }

    // the tree edges will be any edges in the tree, i.e. the pairing of each element
    // and its parent in the tree
    return parlay::tabulate((m_plus_n), [&] (size_t i) {
        // given vertex i, we look for any vertex that is either i's parent 
        // or has i as a parent
        return parlay::filter(parlay::iota<size_t>(m_plus_n), [&] (size_t j) {
            return (parents[j] == i || parents[i] == j) && (j != i);
        });
    });
}

Path euler_tour_from_tree(AdjacencyMatrix &tree_edges, SuccessionMatrix &next_matrix) {
    // we start from 0 as a root, and get the first edge curr1, curr2.
    size_t curr1 {0};
    size_t curr2 {tree_edges[0][0]};
    // save first edge
    size_t first {curr1};
    size_t second {curr2};
    // ahead will be the third vertex, which is the end of the edge after our current edge
    size_t ahead {next_matrix[curr1][curr2].value()};

    Path tour(1, first);
    tour.push_back(second);
    // until we are back at the beginning of the cycle,
    // add the next element and rotate the values.
    // speed up with list ranking may be possible
    while(curr2 != first || ahead != second){
        curr1 = curr2;
        curr2 = ahead;
        tour.push_back(curr2);
        ahead = next_matrix[curr1][curr2].value();
    }

    return tour;
}

Path path_from_euler_tour(Path &tour, size_t left_element, size_t right_element,
                          size_t &start_index, size_t &end_index){
    // get all locations of each element and then the highest index of each.
    parlay::sequence<int> left_element_locations = parlay::tabulate(tour.size(), [&] (size_t i) {return  (tour[i] == left_element)?(int) i:-1;});
    parlay::sequence<int> right_element_locations = parlay::tabulate(tour.size(), [&] (size_t i) {return  (tour[i] == right_element)?(int) i:-1;});
    size_t highest_left_element = static_cast<size_t> (parlay::max_element(left_element_locations) - left_element_locations.begin());
    size_t highest_right_element = static_cast<size_t> (parlay::max_element(right_element_locations) - right_element_locations.begin());

    // we want to get a path from one element to the other. So, we can just go from the lower of the
    // highest elements to the other.
    start_index = (highest_right_element > highest_left_element) ? highest_left_element : highest_right_element;
    end_index = highest_right_element + highest_left_element - start_index;
    size_t low_element {(highest_right_element > highest_left_element) ? left_element : right_element};

    // grab the visit and add on the lower element again to create a cycle
    Path visit {parlay::sequence<size_t>(tour.begin() + start_index, tour.begin() + end_index + 1)};
    visit.push_back(low_element);
    return visit;
}

void push_path_segment(parlay::sequence<parlay::sequence<size_t>> &new_path_segments,
                       Path path, size_t start_index, size_t end_index) {
    // add the sement of the path between given indices to the path segments, as long as it is possible
    if(end_index > start_index)
        new_path_segments.push_back(parlay::sequence<size_t>(path.begin() + start_index, path.begin() + end_index));
}

void recreate_cycle_without_edge(Path &cycle, SuccessionMatrix &next_matrix,
                                 Path &visited_path, size_t edge_index,
                                 size_t path_start, size_t path_end) {
    // first we need to find where the edge in the visited path is located in the cycle
    // because the visited path has been trimmed
    auto iiota = parlay::iota<size_t>(cycle.size()-1);
    size_t forward_location = static_cast<size_t> (find_if(iiota, [&](size_t x) {
        return cycle[x] == visited_path[edge_index] && cycle[x+1] == visited_path[edge_index+1];
    }) - iiota.begin());
    // we also need to find where the reverse edge is in the cycle
    size_t backward_location = static_cast<size_t> (find_if(iiota,[&](size_t x) {
        return cycle[x+1] == visited_path[edge_index] && cycle[x] == visited_path[edge_index+1];
    }) - iiota.begin());

    // lets call u,v the forward edge for clarity in documentation
    parlay::sequence<parlay::sequence<size_t>> new_path_segments;
    
    // if the edge is at the end of the visited path, then we know that the
    // edge we are trying to add in (which is the last two elements of visited path)
    // and the edge being removed (edge_index) are the same, so no need to do anything
    if(edge_index < visited_path.size()-2) {
        // forward location is in the visited path and backward is not
        // we need to distinguish whether backward is before/after the visited path
        if(backward_location < forward_location){
            // this option is ... vb,ub ... path( ... uf,vf ... ) ...
            // transform into ... vb/path(vf ...), path(... uf)/ub ... skip path ...

            // from beginning to v in the v,u edge
            push_path_segment(new_path_segments, cycle, 0,                      backward_location+1);
            // after v in u,v to where visited path ends
            push_path_segment(new_path_segments, cycle, forward_location+2,     path_end+1);
            // from where visited path begins to u in u,v
            push_path_segment(new_path_segments, cycle, path_start,             forward_location+1);
            // from u in v,u edge to start of visited path
            push_path_segment(new_path_segments, cycle, backward_location+2,    path_start+1);
            // from end of visited path to end of cycle
            push_path_segment(new_path_segments, cycle, path_end,               cycle.size());
        } else {
            // this option is ... path( ... uf,vf ... ) ... vb,ub ...
            // transform into ... skip path ... vb/path(vf ...), path(... uf)/ub ...

            // from beginnning to where visited path begins
            push_path_segment(new_path_segments, cycle, 0,                      path_start+1);
            // from end of visited path to v in v,u edge
            push_path_segment(new_path_segments, cycle, path_end,               backward_location+1);
            // from v in u,v edge to end of visited path
            push_path_segment(new_path_segments, cycle, forward_location+2,     path_end+1);
            // from start of visited path to u in u,v
            push_path_segment(new_path_segments, cycle, path_start,             forward_location+1);
            // from u in v,u to end of cycle
            push_path_segment(new_path_segments, cycle, backward_location+2,    cycle.size());
        }
        // we end up with the rounded edges in the tree replaced by an edge between the
        // vertices that begin and end the visited path 

        // remove the edge that was rounded away
        next_matrix[visited_path[edge_index+1]][visited_path[edge_index]] = std::nullopt;
        next_matrix[visited_path[edge_index]][visited_path[edge_index+1]] = std::nullopt;

        // replace the edge with the desired edge which we can get from the first
        // and second-last elements of the visited path
        // The value doesnt matter as we currently use this only as boolean
        next_matrix[visited_path[0]][visited_path[visited_path.size()-2]] = 1;
        next_matrix[visited_path[visited_path.size()-2]][visited_path[0]] = 1;

        // replace the cycle with the new segments stitched together
        cycle = parlay::flatten(new_path_segments);
    }
}

Path round_all_cycles(DoubleMatrix &array, EdgeIndicators &edges){
    size_t m {array.size()};
    size_t n {array[0].size()};

    //convert edge matrix to adjacency matrix
    AdjacencyMatrix adj_matrix {adjacency_matrix_from_edges(edges)};

    //get a tree based on parlaylib BFS.h
    AdjacencyMatrix tree_edges {tree_edges_from_adj_matrix(adj_matrix)};

    // this succession matrix is created in order to be able to see whether an edge
    // is a tree edge faster.
    // see https://en.wikipedia.org/wiki/Euler_tour_technique 
    // this process has possible speedup opportunity using different data structure.
    // (maybe hashmap or list of hashmap)
    SuccessionMatrix next(m+n, parlay::sequence<std::optional<size_t>>(m+n,std::nullopt));

    parlay::parallel_for(0,m+n, [&](size_t u){
        parlay::parallel_for(0,tree_edges[u].size(), [&](size_t v) {
            // we have an established order of tree_edges[u] which is the list of 
            // all edges out of u. So, for each edge into u we set the next value
            // as the next edge in order out of u.
            next[tree_edges[u][v]][u] = tree_edges[u][(v+1) % tree_edges[u].size()];
        });
    });

    // and we immediately use the succession matrix to generate a cycle which
    // traverses the entire tree using both directed edges for all edges.
    Path tour{euler_tour_from_tree(tree_edges, next)};

    // now, we will search through every non-tree edge and round a cycle including
    // that edge, until all edges remaining are in the tree
    for(size_t u {0}; u < m; u++){
        for(size_t v: adj_matrix[u]) {
            // we check that this edge is not in the tree, otherwise skip
            if(!next[u][v].has_value()){
                size_t path_start; 
                size_t path_end;
                // get a path from u to v
                parlay::sequence<size_t> visit{path_from_euler_tour(tour, u, v, path_start, path_end)};
                // trim to minimal path
                trim_path(visit);
                // perform a rounding operation on the path
                size_t i = round_path(array, visit);
                // remove the rounded edge and add u,v as an edge in instead
                recreate_cycle_without_edge(tour, next, visit, i, path_start, path_end);
            }
        }
    }

    return tour;
}

void round_tree(DoubleMatrix &array, Path &tour){
    size_t m {array.size()};
    size_t n {array[0].size()};
    size_t len {tour.size()};

    // if the path is a single element, done
    if(len < 2)
        return;

    // Note leafs are easily indicated by a directed edge being
    // immediately followed by its opposite direction edge, i.e. tour[i] == tour[i+2]
    parlay::sequence<size_t> first {parlay::tabulate(len-2, [&] (size_t i) {
        return  (tour[i] == tour[i+2])? i : len;
    })};
    // get first leaf
    size_t first_leaf_index {static_cast<size_t> (parlay::min_element(first)+1 - first.begin())};

    // find a leaf that must be after the first one
    parlay::sequence<size_t> second {parlay::tabulate(len-1, [&] (size_t i) {
        return  (tour[i] == tour[i+2] && i >= first_leaf_index)? i : 2*len-i;
    })};
    // second leaf
    size_t second_leaf_index {static_cast<size_t> (parlay::min_element(second) - second.begin()+1)};

    // from first to second leaf will be our path
    Path visit(tour.begin() + first_leaf_index, tour.begin() + second_leaf_index + 1);

    // round the path and get location of forward edge in the path
    size_t forward_location {round_path(array, visit)};
    // get location of backwards edge in tour
    size_t backward_location;
    parlay::parallel_for(0, tour.size()-1, [&] (size_t x){
        if(tour[x+1] == visit[forward_location] && tour[x] == visit[forward_location+1])
            backward_location = x;
    });
    // get location of forward edge in tour instead of path
    forward_location += first_leaf_index;

    // determine order of forward and backward edges.
    size_t first_location {(backward_location < forward_location)? backward_location : forward_location};
    size_t second_location {backward_location + forward_location - first_location};
    // split the tree. first tree will be between the two edges
    Path tree_1(tour.begin() + first_location+1, tour.begin()+ second_location+1);
    // second tree will be after and before first tree combined
    Path tree_2(tour.begin() + second_location+1, tour.end()-1);
    tree_2.append(tour.begin(), tour.begin()+first_location+1);
    // round both subtrees in parallel
    parlay::par_do(
        [&]() {round_tree(array, tree_1);},
        [&]() {round_tree(array, tree_2);}
    );
}

extern "C" void dependent_rounding(double *array, size_t m, size_t n, size_t seed, size_t logging_level){
    srand(seed);

    // set up logging
    std::chrono::_V2::system_clock::time_point start_time;
    std::chrono::_V2::system_clock::time_point stage_start_time;
    long num_edges_remaining;
    long last_stage_num_edges_remaining;
    long long stage_time_elapsed;
    long long total_time_elapsed;
    if(logging_level) {
        start_time = std::chrono::high_resolution_clock::now();
        stage_start_time = start_time;
    }
        
    // grab from array to put into parlaylib data structures
    // array is 1D because that is how it is passed
    DoubleMatrix parlay_array(m, parlay::sequence<double>(n));
    parlay::parallel_for(0,m, [&](size_t i){
        parlay::parallel_for(0,n, [&](size_t j) {
            parlay_array[i][j] = array[i*n+j];
        });
    });

    // get edge indicator matrix (edge is already rounded if it has integer value)
    EdgeIndicators edges(m, parlay::sequence<bool>(n, true));
    round_all_butterflies(parlay_array, edges);

    if(logging_level) {
        stage_time_elapsed = get_ms_elapsed(stage_start_time);
        num_edges_remaining = parlay::scan(parlay::map(parlay_array, [&](auto e) {
            return parlay::scan(parlay::map(e, [&](double v) {
                return (v != floor(v))? 1: 0;
            })).second;
        })).second;
        std::cout << "Butterfly rounding completed\n";
        std::cout << "Edges rounded: " << m*n - num_edges_remaining << "\n";
        std::cout << "Time elapsed (ms): " << stage_time_elapsed << "\n";
        std::cout << "Avg time per 1000 rounded elements (ms): " << static_cast<double>(stage_time_elapsed) * 1000.0 / (m*n - num_edges_remaining) << "\n" << "\n";
        last_stage_num_edges_remaining = num_edges_remaining;
        stage_start_time = std::chrono::high_resolution_clock::now();
    }
    

    // need to get the Euler tour after rounding cycles
    Path tour;
    tour = round_all_cycles(parlay_array, edges);

    if(logging_level) {
        stage_time_elapsed = get_ms_elapsed(stage_start_time);
        num_edges_remaining = parlay::scan(parlay::map(parlay_array, [&](auto e) {
            return parlay::scan(parlay::map(e, [&](double v) {
                return (v != floor(v))? 1: 0;
            })).second;
        })).second;
        std::cout << "Cycle rounding completed\n";
        std::cout << "Edges rounded: " << last_stage_num_edges_remaining - num_edges_remaining << "\n";
        std::cout << "Time elapsed (ms): " << stage_time_elapsed << "\n";
        std::cout << "Avg time per 1000 rounded elements (ms): " << static_cast<double>(stage_time_elapsed) * 1000.0 / (last_stage_num_edges_remaining - num_edges_remaining) << "\n" << "\n";
        last_stage_num_edges_remaining = num_edges_remaining;
        stage_start_time = std::chrono::high_resolution_clock::now();
    }

    round_tree(parlay_array, tour);
    parlay::parallel_for(0,m, [&](size_t i){
        parlay::parallel_for(0,n, [&](size_t j) {
            // all values will be very near an integer, but to ensure problems
            // with floating values have been removed, do a final round.
            array[i*n+j] = parlay_array[i][j];
        });
    });
    
    // finish with printing
    if(logging_level) {
        stage_time_elapsed = get_ms_elapsed(stage_start_time);
        total_time_elapsed = get_ms_elapsed(start_time);
        std::cout << "Tree rounding completed\n";
        std::cout << "Edges rounded: " << last_stage_num_edges_remaining << "\n";
        std::cout << "Time elapsed (ms): " << stage_time_elapsed << "\n";
        std::cout << "Avg time per 1000 rounded elements (ms): " << static_cast<double>(stage_time_elapsed) * 1000.0 / last_stage_num_edges_remaining << "\n" << "\n";

        std::cout << "Total time elapsed (ms): " << total_time_elapsed << "\n";
        std::cout << "Overall avg time per 1000 rounded elements (ms): " << static_cast<double>(total_time_elapsed) * 1000.0 / m / n << "\n";
    }
}
