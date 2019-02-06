import numpy as np


def standardize(x):
    """standardize data.

    x: input data
    """
    return (X - np.mean(X)) / np.std(X)
