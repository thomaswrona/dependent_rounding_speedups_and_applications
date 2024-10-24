/**
 * @file
 * @author  Thomas Wrona <tomdwrona@gmail.com>
 *
 * @section DESCRIPTION
 *
 * This file contains a stochastic rounding function mostly for comparison to dependent rounding.
 */

#ifndef STOCHASTIC_ROUND_H
#define STOCHASTIC_ROUND_H

#include <parlay/primitives.h>

/**
* Function that takes an m x n array and rounds each element stochastically.
* 
* The probability that an element is rounded down is equal to its distance to 
* its ceiling, and the probability of rounding up is equal to its distance to
* its floor. This ensures that the rounded value is unbiased. 
*
* @param array is the input array to round size mn (flattened 2d m x n array).
*        It is rounded in place.
*
* @param m is the first dimension size.
*
* @param n is the second dimension size.
* 
* @param seed is the random seed used.
*/
extern "C" void stochastic_rounding(double *array, size_t m, size_t n, size_t seed);

#endif