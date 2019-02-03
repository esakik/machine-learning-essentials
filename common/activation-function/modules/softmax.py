import numpy as np

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
