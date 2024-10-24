/**
 * @file
 * @author  Thomas Wrona <tomdwrona@gmail.com>
 *
 * @section DESCRIPTION
 *
 * This file contains a standard rounding function mostly for comparison to dependent rounding.
 */

#include <iostream>
#include <cmath>

#include <parlay/primitives.h>

#include "round_util.h"
#include "standard.h"

extern "C" void standard_rounding(double *array, size_t m, size_t n) {
    // for each element in parallel
    parlay::parallel_for(0,m, [&](size_t i){
        parlay::parallel_for(0,n, [&](size_t j) {
            // basic rounding on each element
            array[i*n+j] = round(array[i*n+j]);
        });
    });
}