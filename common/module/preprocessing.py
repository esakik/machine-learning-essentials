import numpy as np


def standardize(x):
    """standardize data.

    x: input data
    """
    return (x - np.mean(x)) / np.std(x)
