import numpy as np


def standardize(x):
    """standardize data.

    x: input data
    """
    return (x - np.mean(x, axis=0)) / np.std(x, axis=0)
