import numpy as np

def sigmoid(x):
    """Sigmoid function.
    
    x: input data
    """
    return 1 / (1 + np.exp(-x))
