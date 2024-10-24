/**
 * @file
 * @author  Thomas Wrona <tomdwrona@gmail.com>
 *
 * @section DESCRIPTION
 *
 * This file contains a standard rounding function mostly for comparison to dependent rounding.
 */

#ifndef STANDARD_ROUND_H
#define STANDARD_ROUND_H

#include <parlay/primitives.h>

/**
* Function that takes an m x n array and rounds each element independently.
*
* @param array is the input array to round size mn (flattened 2d m x n array).
*        It is rounded in place.
*
* @param m is the first dimension size.
*
* @param n is the second dimension size.
*/
extern "C" void standard_rounding(double *array, size_t m, size_t n);

#endif