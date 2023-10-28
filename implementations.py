import numpy as np

def least_squares(y, tx): # Calculates the least squares solution using normal equations.
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    e = y - tx @ w
    loss = 0.5 * np.mean(e**2)
    
    return w, loss

def ridge_regression(y, tx, lambda_): # Calculates the ridge regression solution using normal equations.

    a = tx.T @ tx + lambda_ * np.eye(tx.shape[1])
    b = tx.T @ y
    w = np.linalg.solve(a, b)
    e = y - tx @ w
    loss = 0.5 * np.mean(e**2)
    
    return w, loss
