# Dependent Rounding Speedups and Applications

Code supporting Thomas Wrona's master's thesis at http://hdl.handle.net/1903/32697

The original idea for this project was to find applications of the dependent rounding algorithm https://www.cs.umd.edu/~samir/grant/jacm06.pdf to machine learning. However, a naievely implemented dependent rounding algorithm in Python can take upwards of 30 minutes on matrices size 1000 by 1000. The main issue with the naieve algorithm is that it repeatedly calls DFS on a bipartite graph to find cycles, instead of either explicitly looking for small cycles or saving information for use in the next DFS call. This new algorithm remedies this, bringing the time to round a matrix size 1000 by 1000 down to a few seconds in Python or less than one second in C++. This was the bulk of the work for this project.

Some example machine learning applications (based on the Lottery Ticket Hypothesis and signSGD) are also presented here. The speedup for dependent rounding certainly allows us to use it in scenarios it is called every epoch or less frequently (Lottery Ticket Hypothesis), though it still is not likely feasible to use every iteration (signSGD). It may be possible if the entire algorithm is implemented in CUDA or Tensorflow graph mode (or similar) and/or with enough processing units.

The primary case for dependent rounding is random assignment (e.g. scheduling) for which this can also be used.

## SETUP EXAMPLE FOR WSL

1. Clone

    git clone https://github.com/thomaswrona/dependent_rounding_speedups_and_applications.git

2. Get dependencies for C++ code

    Install ParlayLib via https://cmuparlay.github.io/parlaylib/installation.html

3. Create shared object

    Use command similar to (can change desired compiler, use -Os instead of -O3, etc.):
    g++ -pthread -std=c++17 -g ./cpp_src/round_util.cpp ./cpp_src/dependent.cpp ./cpp_src/standard.cpp ./cpp_src/stochastic.cpp -shared -fPIC -O3 -o ./py_src/dependent.so

4. Get dependencies for Python module

    pip install numpy

5. Install Python module

    pip install ./py_src
    or, for editing mode,
    pip install -e ./py_src

6. Test

    Easiest way to test is to run one of the Jupyter notebooks.


## STRUCTURE

'''.
+-- cpp_src                                         # Source for C++ rounding functions
|   +-- dependent.cpp                               # Dependent rounding C++
|   +-- dependent.h                                 #
|   +-- hash_map.h                                  # Hashmap implementation from ParlayLib examples
|   +-- round_util.cpp                              # Util functions
|   +-- round_util.h                                #
|   +-- standard.cpp                                # Standard rounding C++, mostly for comparison
|   +-- standard.h                                  #
|   +-- stochastic.cpp                              # Stochastic rounding C++, mostly for comparison
|   +-- stochastic.h                                #
+-- ml_example                                      # Machine learning examples, see README in this folder for more info
+-- notebooks                                       # Jupyter notebook demonstrations
|   +-- All_Rounding_Demonstrations.ipynb           # Basic demonstration of rounding functions
|   +-- Dependent_Rounding_Demonstration.ipynb      # Shows basic properties of dependent rounding
|   +-- Standard_Rounding_Demonstration.ipynb       # Shows basic properties of standard rounding
|   +-- Stochastic_Rounding_Demonstration.ipynb     # Shows basic properties of stochastic rounding
|   +-- Timing_Demonstrations.ipynb                 # Has time comparisons of rounding methods
+-- py_src                                          # source for Python rounding functions
|   +-- __init__.py                                 # init file for module
|   +-- dependent_rounding.py                       # Python rounding functions and C++ wrapper
|   +-- dependent_sparsification.py                 # Python functions to create 0/1 mask via rounding for sparsification
|   +-- setup.py                                    # setup file for module
+-- .gitignore                                      # gitignore
+-- LICENSE.txt                                     # MIT License
+-- README.md                                       # README'''