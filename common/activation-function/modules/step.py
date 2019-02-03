import numpy as np

def step(x):
    """Step function.
    
    x: input data
    """
    return np.array(x > 0, dtype=np.int)
