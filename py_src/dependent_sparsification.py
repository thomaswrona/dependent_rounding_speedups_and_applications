from dependent_rounding import round_matrix
import numpy as np

def get_quantiles(layer):
    """Generates quantile array for a layer (i.e. values that represent what
    percentage of nonzero elements are below each element). Quantile of 0 is 0.

    Args:
        layer (numpy float array): layer to get quantiles for

    Returns:
        numpy float array: quantiles array in same shape
    """
    sorted_indices = np.argsort(layer, axis = None) #sort all values in any order
    num_nonzero = np.count_nonzero(layer) #ignore zeros

    # now reformat quantiles to go from smallest to largest nonzero elements
    quantiles = (sorted_indices - (layer.size - num_nonzero)) / (num_nonzero - 1)
    quantiles[layer == 0] = 0
    return quantiles

def generate_probability(layer, distribution_method, use_quantiles, sparsification_level):
    """Generate probabilities to use with probabilistic rounding for a layer.

    Args:
        layer (numpy float array): layer to get probabilities for.
        distribution_method (str): "exp" means an exponential distribution is used.
                                   "sig" means a sigmoid distribution is used.
                                   Any other value results in a linear distribution.
        use_quantiles (bool): Whether to use quantiles (True) or layer
                              values (False) to generate probabilites.
        sparsification_level (float): What percentage of layer to remove,
                                      between 0 and 1.

    Returns:
        numpy float array: probability array in same shape.
    """
    if use_quantiles:
        quantiles = get_quantiles(np.abs(layer))
    else:
        quantiles = np.abs(layer)

    if(distribution_method == "exp"):
        # exponential
        return np.power(quantiles, sparsification_level / (1 - sparsification_level))
    elif(distribution_method == "sig"):
        # sigmoid
        return 1 / (1 + np.exp(-(quantiles-sparsification_level)/(sparsification_level*(1-sparsification_level)*0.25)))
    else:
        # linear distribution capped between 0 and 1
        return np.maximum(0.0,np.minimum(1.0,2*quantiles*(1 - sparsification_level)))

def standard_sparsification(layer, sparsification_level = 0.5):
    """Standard / absolute value sparsification. Low magnitude elements are removed.

    Args:
        layer (numpy float array): layer to sparsify
        sparsification_level (float, optional): What percentage of layer to remove,
                                                between 0 and 1. Defaults to 0.5.

    Returns:
        numpy float array: sparsified layer
    """
    # get value at the sparsification level quantile (i.e. x% are below this value)
    quantile_level = np.quantile(np.abs(layer), sparsification_level)
    # swap low magnitude values out for 0
    return np.where((layer > quantile_level) | (layer < -quantile_level), layer, 0)

def stochastic_sparsification(layer, distribution_method = None,
                              use_quantiles = False, sparsification_level = 0.5):
    """Stochastic sparsification. Lower magnitude elements have a higher chance
    of being removed (chance is inversely proportional to magnitude).

    Args:
        layer (numpy float array): layer to sparsify
        distribution_method (str, optional): "exp" means an exponential distribution is used.
                                             "sig" means a sigmoid distribution is used.
                                             Any other value results in a linear distribution.
                                             Defaults to None.
        use_quantiles (bool, optional): Whether to use quantiles (True) or layer
                                        values (False) to generate probabilites.
                                        Defaults to False.
        sparsification_level (float, optional): What percentage of layer to remove,
                                                between 0 and 1. Defaults to 0.5.

    Returns:
        numpy float array: sparsified layer
    """
    # get probabilities, round to get 0/1 mask, then multiply by layer to get values
    return round_matrix(generate_probability(layer,
                                             distribution_method,
                                             use_quantiles,
                                             sparsification_level),
                        method = "stochastic") * layer

def dependent_sparsification(layer, distribution_method = None,
                             use_quantiles = False, sparsification_level = 0.5,
                             keep_weights = False):
    """Dependent sparsification. Lower magnitude elements have a higher chance
    of being removed (chance is inversely proportional to magnitude).

    Args:
        layer (numpy float array): layer to sparsify
        distribution_method (str, optional): "exp" means an exponential distribution is used.
                                             "sig" means a sigmoid distribution is used.
                                             Any other value results in a linear distribution.
                                             Defaults to None.
        use_quantiles (bool, optional): Whether to use quantiles (True) or layer
                                        values (False) to generate probabilites.
                                        Defaults to False.
        sparsification_level (float, optional): What percentage of layer to remove,
                                                between 0 and 1. Defaults to 0.5.
        keep_weights (bool, optional): Whether to keep weight values intact or
                                       normalize to make result unbiased. Defaults to False.

    Returns:
        numpy float array: sparsified layer
    """
    probs = generate_probability(layer, distribution_method, use_quantiles, sparsification_level)
    # round to get 0/1 mask, then multiply by layer or normalized layer (latter for unbiased)
    return round_matrix(np.copy(probs),
                        method = "dependent") * (layer if keep_weights else layer / probs)

def sparsify_weights(weights, sparsification_level, sparse_method = "dependent",
                     disribution_method = None, use_quantiles = False,
                     keep_weights = False, separate_by_sign = False):
    """Perform chosen sparsification function on all weights.

    Args:
        weights (list of numpy float arrays): weights to be sparsified.
        sparsification_level (float, optional): What percentage of layer to remove,
                                                between 0 and 1. Defaults to 0.5.
        sparse_method (str, optional): Which method to use: "standard", "stochastic",
                                       or "dependent". Defaults to "dependent".
        distribution_method (str, optional): "exp" means an exponential distribution is used.
                                             "sig" means a sigmoid distribution is used.
                                             Any other value results in a linear distribution.
                                             Defaults to None.
        use_quantiles (bool, optional): Whether to use quantiles (True) or layer
                                        values (False) to generate probabilites.
                                        Defaults to False.
        keep_weights (bool, optional): Whether to keep weight values intact or
                                       normalize to make result unbiased. Defaults to False.
        separate_by_sign (bool, optional): Whether to separate layers into positive/negative
                                           before sparsifying. Defaults to False.

    Returns:
        list of numpy float arrays: sparsified weights.
    """
    
    # get sparsification function
    if sparse_method == "standard":
        sparse_function = lambda layer: standard_sparsification(layer,
                                                                sparsification_level)
    elif sparse_method == "stochastic":
        sparse_function = lambda layer: stochastic_sparsification(layer,
                                                                  disribution_method,
                                                                  use_quantiles,
                                                                  sparsification_level)
    else:
        sparse_function = lambda layer: dependent_sparsification(layer,
                                                                 disribution_method,
                                                                 use_quantiles,
                                                                 sparsification_level,
                                                                 keep_weights)

    # call sparsification function on each layer, or on positive/negative elements separately
    return [sparse_function(l[l>0]) - sparse_function(-l[l<0]) if separate_by_sign else sparse_function(l) for l in weights]