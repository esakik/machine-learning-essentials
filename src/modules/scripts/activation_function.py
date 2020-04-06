import numpy as np


def step(x):
    """Step function.

    x: input data
    """
    return np.array(x > 0, dtype=np.int)


def sigmoid(x):
    """Sigmoid function.

    x: input data
    """
    return 1 / (1 + np.exp(-x))


def relu(x):
    """ReLU function.

    x: input data
    """
    return np.maximum(0, x)


def softmax(x):
    """Softmax function.

    x: input data
    """
    if x.ndim == 2:
        x = x.T - np.max(x.T, axis=0)
        return (np.exp(x) / np.sum(np.exp(x), axis=0)).T

    # Prevent overflow
    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))
