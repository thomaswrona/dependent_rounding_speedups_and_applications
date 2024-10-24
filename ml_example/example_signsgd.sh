#!/bin/bash

#python example.py --experiment_type depsgd --iterations 3 --epochs 50 --which_set mnist --prune_strategy momentum
#python example.py --experiment_type depsgd --iterations 3 --epochs 50 --which_set mnist --prune_strategy sign
#python example.py --experiment_type depsgd --iterations 1 --epochs 50 --which_set mnist --prune_strategy stoch
##python example.py --experiment_type depsgd --iterations 1 --epochs 50 --which_set mnist --prune_strategy dep

#python example.py --experiment_type depsgd --iterations 3 --epochs 50 --which_set cifar10 --prune_strategy momentum
#python example.py --experiment_type depsgd --iterations 3 --epochs 50 --which_set cifar10 --prune_strategy sign
#python example.py --experiment_type depsgd --iterations 1 --epochs 50 --which_set cifar10 --prune_strategy stoch
##python example.py --experiment_type depsgd --iterations 1 --epochs 50 --which_set cifar10 --prune_strategy dep

# note: do not recommend using dep prune_strategy, it is very slow (need to rewrite dependent rounding in cuda/tf cpp builtins)