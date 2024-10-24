/**
 * @file
 * @author  Thomas Wrona <tomdwrona@gmail.com>
 *
 * @section DESCRIPTION
 *
 * This file contains functions that are general utility and are not implicitly tied to one rounding method.
 */

#ifndef ROUND_UTIL_H
#define ROUND_UTIL_H

#include <chrono>
#include <tuple>

/**
* Function that generates a single pairing in the style of a round-robin tournament.
*
* It functions by fixing the first "player" and rotating the remaining "players"
* every round. The purpose is to be able to get all possible pairwise assignments
* in a disjoint method as fast as possible. Note that for an odd number of players,
* we will be treating it as if there is one more player and granting one player
* each round a "bye" where that pair will not be used.
* 
* See https://en.wikipedia.org/wiki/Round-robin_tournament circle method.
*
* @param pair_index is the current pair number to generate. Between 0 and
*        (rounds-1)//2 inclusive
*
* @param current_round is the current round number to generate for
*        between 0 and rounds-1 inclusive
*
* @param rounds is the total number of rounds which will be generated
*        for n "players", it should be n-1 if n is even and n if n is odd.
*
* @return a tuple of the pair generated
*/
std::tuple<size_t,size_t> generate_round_robin(size_t pair_index, size_t current_round, size_t rounds);

/**
* Function that takes two values and returns the unique corresponding cantor id.
* 
* It is possible that this has overflow issues for small size_t maximum possible
* values and large values given, but these are currently not feasible.
*
* See https://en.wikipedia.org/wiki/Pairing_function#Cantor_pairing_function.
*
* @param k1 is a first value.
*
* @param k2 is a second value.
*
* @return The cantor pair id associated with this pair of values.
*/
size_t cantor_pair(size_t k1, size_t k2);

/**
* Get the distance of a double from its integer floor.
*
* @param d is the double.
*
* @return difference between d and its integer floor.
*/
double distance_to_floor(double d);

/**
* Get the milliseconds that have elapsed since specified time.
*
* @param start_time is the time to subtract from.
*
* @return milliseconds elapsed since start_time.
*/
long long get_ms_elapsed(std::chrono::_V2::system_clock::time_point start_time);

#endif