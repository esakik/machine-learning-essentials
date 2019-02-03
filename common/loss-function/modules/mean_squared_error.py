import numpy as np

def mean_squared_error(y, t):
    """Calculate mean squared error.
    
    y: predicted data
    t: target data
    """
    return 0.5 * np.sum((y - t)**2)
