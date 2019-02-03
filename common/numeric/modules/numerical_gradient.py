def _numerical_gradient_1d(f, x):
    """Calculate numeric gradient for 1d.

    f: function
    x: input data
    """
    h = 1e-4
    grad = np.zeros_like(x)
    
    for idx in range(x.size):
        tmp_val = x[idx]
        
        # f(x+h)
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) 
        
        # f(x-h)
        x[idx] = tmp_val - h 
        fxh2 = f(x) 
        
        grad[idx] = (fxh1 - fxh2) / (2 * h)
        
        # restore
        x[idx] = tmp_val
        
    return grad


def numerical_gradient(f, X):
    """Calculate numeric gradient for 1d.

    f: function
    X: input data
    """
    if X.ndim == 1:
        return _numerical_gradient_1d(f, X)
    else:
        grad = np.zeros_like(X)
        
        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_1d(f, x)
        
        return grad
