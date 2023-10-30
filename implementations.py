import numpy as np


def remove_empty_columns(x_train, x_test, threshold):
    """
    Remove columns who contain more than threshold% NaN values
    """
    is_nan_arr = np.isnan(x_train)
    #percentage of missing values
    missing_percentage = np.mean(is_nan_arr, axis=0)
    #threshold of NaN values
    #columns to keep
    to_keep = missing_percentage < threshold
    return x_train[:, to_keep], x_test[:, to_keep]


def standardize_data (x_train, x_test):
    x_train_array = x_train.copy()
    for col_idx in range(x_train_array.shape[1]):
        unique_values = np.unique(x_train_array[~np.isnan(x_train_array[:, col_idx]), col_idx])
        if len(unique_values) == 1 and 1 in unique_values:
            x_train_array[np.isnan(x_train_array[:, col_idx]), col_idx] = 0
    x_test_array = x_test.copy()
    for col_idx in range(x_test_array.shape[1]):
        unique_values = np.unique(x_test_array[~np.isnan(x_test_array[:, col_idx]), col_idx])
        if len(unique_values) == 1 and 1 in unique_values:
            x_test_array[np.isnan(x_test_array[:, col_idx]), col_idx] = 0

    mean_x_train = np.nanmean(x_train_array, axis = 0)
    std_x_train = np.nanstd(x_train_array, axis = 0)
    std_x_train += 1e-10
    x_train_standardized = (x_train_array - mean_x_train) / std_x_train
    x_train_standardized[np.isnan(x_train_standardized)] = 0
    x_test_standardized = (x_test_array - mean_x_train) / std_x_train
    x_test_standardized[np.isnan(x_test_standardized)] = 0
    return x_train_standardized, x_test_standardized


def sigmoid(t):
    """apply sigmoid function on t.

    Args:
        t: scalar or numpy array

    Returns:
        scalar or numpy array
    """
    return 1.0 / (1 + np.exp(-t))


def compute_gradient_mse(y, tx, w):
    """
    Compute the gradient for mean squared error.
    """
    e = y - tx.dot(w)
    gradient = -tx.T.dot(e) / len(y)
    return gradient

def compute_loss_mae(err):
    """
    Compute the loss for mean absolute error.
    """
    return np.mean(np.abs(err))


def compute_subgradient_mae(y, tx, w):
    """
    Compute the gradient for mean squared error.
    """
    err = y - tx.dot(w)
    grad = -np.dot(tx.T, np.sign(err)) / len(err)
    return grad, err

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """
    Calculate MSE for GD
    """
    w = initial_w
    for _ in range(max_iters):
        gradient = compute_gradient_mse(y, tx, w)
        w = w - gamma * gradient
    loss = 1 / 2 * np.mean((y - tx.dot(w))**2)
    return w, loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """
    Calculate MSE for SGD
    """
    w = initial_w
    for _ in range(max_iters):
        i = np.random.randint(0, len(y))
        gradient = compute_gradient_mse(y[i : i + 1], tx[i : i + 1], w)
        w = w - gamma * gradient
    loss = 1 / 2 * np.mean((y - tx.dot(w))**2)
    return w, loss

def mean_absolute_error_gd(y, tx, initial_w, max_iters, gamma):
    """
    Calculate MAE for GD
    """
    w = initial_w
    for n_iter in range(max_iters):
        grad, err = compute_subgradient_mae(y, tx, w)
        w = w - gamma * grad
    loss = compute_loss_mae(err)
    return w, loss


def mean_absolute_error_sgd(y, tx, initial_w, max_iters, gamma):
    """
    Calculate MAE for SGD
    """
    w = initial_w
    for _ in range(max_iters):
        i = np.random.randint(0, len(y))
        grad, err = compute_subgradient_mae(y[i : i + 1], tx[i : i + 1], w)
        w = w - gamma * grad
    loss = np.mean(np.abs(y - tx.dot(w)))
    return w, loss


def compute_gradient_logistic(y, tx, w):
    """
    Compute the gradient for logistic regression.
    """
    prediction = sigmoid(tx @ w)
    gradient = tx.T @ (prediction - y) / len(y)
    return gradient


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
    """calculate the least squares."""
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    e = y - tx @ w
    loss = 0.5 * np.mean(e ** 2)

    return w, loss


def ridge_regression(y, tx, lambda_):
    """implement ridge regression.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
    """

    aI = 2 * tx.shape[0] * lambda_ * np.eye(tx.shape[1])
    a = tx.T @ tx + aI
    b = tx.T @ y
    w = np.linalg.solve(a, b)

    e = y - tx @ w
    loss = 0.5 * np.mean(e ** 2)
    return w, loss

def compute_f1_score(true_labels, predictions) :
    """
    Compute the F1 score based on true labels and predictions.
    """
    TP = np.sum((predictions == 1) & (true_labels == 1))
    FP = np.sum((predictions == 1) & (true_labels == -1))
    FN = np.sum((predictions == -1) & (true_labels == 1))
    
    precision = TP / (TP + FP) if TP + FP != 0 else 0
    recall = TP / (TP + FN) if TP + FN != 0 else 0
    
    f1_score = (2 * precision * recall) / (precision + recall) if precision + recall != 0 else 0
    return f1_score

def train_test_split(y, tx):
    """
    Splits the dataset into training and test sets.
    """
    test_ratio = 0.2
    shuffled_indices = np.random.permutation(tx.shape[0])

    test_size = int(test_ratio * tx.shape[0])
    test_indices = shuffled_indices[:test_size]
    train_indices = shuffled_indices[test_size:]

    x_train = tx[train_indices]
    y_train = y[train_indices]
    x_test = tx[test_indices]
    y_test = y[test_indices]
    return x_train, y_train, x_test, y_test