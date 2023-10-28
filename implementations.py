import numpy as np


def compute_gradient_mse(y, tx, w):
    """
    Compute the gradient for mean squared error.
    """
    y = y.reshape(-1, 1)
    e = y - tx @ w
    gradient = -tx.T @ e / len(y)
    return gradient


def sigmoid(t):
    """
    simple sigmoid function
    """
    return 1.0 / (1 + np.exp(-t))


def compute_gradient_logistic(y, tx, w):
    """
    Compute the gradient for logistic regression.
    """
    prediction = sigmoid(tx @ w)
    gradient = tx.T @ (prediction - y) / len(y)
    return gradient


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """
    Calculate MSE for GD
    """
    w = initial_w
    for _ in range(max_iters):
        gradient = compute_gradient_mse(y, tx, w)
        w = w - gamma * gradient
    loss = 0.5 * np.mean((y - tx @ w) ** 2)
    return w, loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """
    Calculate MSE for SGD
    """
    w = initial_w
    for _ in range(max_iters):
        i = np.random.randint(0, len(y))
        gradient = compute_gradient_mse(y[i], tx[i], w)
        w = w - gamma * gradient
    loss = 0.5 * np.mean((y - tx @ w) ** 2)
    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    calculate logistic regression loss
    """
    w = initial_w
    for _ in range(max_iters):
        gradient = compute_gradient_logistic(y, tx, w)
        w = w - gamma * gradient
    loss = np.mean(y * np.log(sigmoid(tx @ w)) + (1 - y) * np.log(1 - sigmoid(tx @ w)))
    return w, -loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Regulatized logistic regression loss
    """
    w = initial_w
    for _ in range(max_iters):
        gradient = compute_gradient_logistic(y, tx, w) + 2 * lambda_ * w
        w = w - gamma * gradient
    loss = np.mean(y * np.log(sigmoid(tx @ w)) + (1 - y) * np.log(1 - sigmoid(tx @ w)))
    return w, -loss


def least_squares(y, tx):
    """
    Calculates the least squares solution using normal equations.
    """
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    e = y - tx @ w
    loss = 0.5 * np.mean(e**2)

    return w, loss


def ridge_regression(y, tx, lambda_):
    """
    Calculates the ridge regression solution using normal equations.
    """

    a = tx.T @ tx + lambda_ * np.eye(tx.shape[1])
    b = tx.T @ y
    w = np.linalg.solve(a, b)
    e = y - tx @ w
    loss = 0.5 * np.mean(e**2)

    return w, loss
