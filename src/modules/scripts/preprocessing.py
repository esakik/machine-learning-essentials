import numpy as np


def standardize(X):
    """standardize data.

    X: input data
    """
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)


def decorrelate(X):
    """decorrelate data.

    X: input data
    """
    Sigma = np.cov(X, rowvar=0)
    eigenvalue, eigenvector = np.linalg.eig(Sigma)
    S = eigenvector

    return np.dot(S.T, X.T).T


def whitening(X):
    """whitening data.

    X: input data
    """
    X_centerized = X - X.mean(axis=0)

    Sigma = np.cov(X, rowvar=0)
    _, S = np.linalg.eig(Sigma)

    Lambda = np.dot(np.dot(np.linalg.inv(S), Sigma), S)
    Lambda_sqrt_inv = np.linalg.inv(np.sqrt(Lambda))

    return np.dot(np.dot(X_centerized, S), Lambda_sqrt_inv.T)
