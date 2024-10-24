/**
 * @file
 * @author  Thomas Wrona <tomdwrona@gmail.com>
 *
 * @section DESCRIPTION
 *
 * This file contains functions that are general utility and are not implicitly tied to one rounding method.
 */

#include <chrono>
#include <tuple>
#include <iostream>
#include <cmath>

#include "round_util.h"

std::tuple<size_t,size_t> generate_round_robin(size_t pair_index, size_t current_round, size_t rounds) {
    return std::make_tuple(
        // The first element is fixed at 0 for pair_index = 0. Otherwise, note that
        // the current_round rotates the remaining values by decrementing.
        // The 2*rounds is to maintain values above 0 and the addition of 1 at the
        // end is a byproduct of using 0 as a fixed point.
        pair_index ? 1 + (2*rounds - current_round - 1 + pair_index) % rounds : 0u,
        // Second element similar, but no special condition for pair_index 0.
        1 + (2*rounds - current_round - 1 - pair_index) % rounds);
}

size_t cantor_pair(size_t k1, size_t k2){
    return (k1+k2)*(k1+k2+1)/2 + k2;
}

double distance_to_floor(double d) { 
    return d - floor(d); 
}

long long get_ms_elapsed(std::chrono::_V2::system_clock::time_point start_time){
    // Simple subtraction of start_time from current time then reformat and print.
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time).count();
}