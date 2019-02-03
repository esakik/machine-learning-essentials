import numpy as np

def relu(x):
    """ReLU function.
    
    x: input data
    """
    return np.maximum(0, x)
