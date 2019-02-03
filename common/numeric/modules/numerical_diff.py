def numerical_diff(f, x):
    """Calculate numerical differentiation.
    
    f: function
    x: input data
    """
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)
