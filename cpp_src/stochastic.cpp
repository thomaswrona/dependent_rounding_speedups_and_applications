/**
 * @file
 * @author  Thomas Wrona <tomdwrona@gmail.com>
 *
 * @section DESCRIPTION
 *
 * This file contains a stochastic rounding function mostly for comparison to dependent rounding.
 */

#include <iostream>
#include <cmath>

#include <parlay/primitives.h>

#include "round_util.h"
#include "stochastic.h"

extern "C" void stochastic_rounding(double *array, size_t m, size_t n, size_t seed) {
    // use parlaylib random generation on std real distribution between 0 and 1
    parlay::random_generator gen(seed);
    std::uniform_real_distribution<double> rand_real(0.0, std::nextafter(1.0, 2.0));

    // for each element in parallel
    parlay::parallel_for(0,m, [&](size_t i){
        parlay::parallel_for(0,n, [&](size_t j) {
            auto r = gen[i*n+j];
            // this indexing is how it is passed using ctypes
            // if the rando double is lower than the distance to floor, then round up
            // otherwise round down.
            array[i*n+j] = (rand_real(r) < array[i*n+j]) ? 1.0 : 0.0;
        });
    });
}