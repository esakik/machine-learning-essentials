import numpy as np

def cross_entropy_error(y, t):
    """Calculate cross entropy error.
    
    y: predicted data
    t: target data
    """
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # convert the target data into index of a correct label if the target is one-hot-vector
    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
