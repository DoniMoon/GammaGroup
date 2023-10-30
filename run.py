from helpers import *
import numpy as np
from implementations import *

#Load the data
datapath = "dataset/"
x_train, x_test, y_train, train_ids, test_ids = load_csv_data(datapath)

#Clean and Standardize the data
x_train_stripped, x_test_stripped = remove_empty_columns(x_train, x_test, 0.6)
x_train_standardized, x_test_standardized = standardize_data(x_train_stripped, x_test_stripped)

w, loss = mean_absolute_error_gd(y_train, x_train_standardized, np.zeros(x_train_standardized.shape[1]), 500, 0.01)

result = x_test_standardized.dot(w)
binary_predictions = np.where(result >= 0.9, 1, -1)
create_csv_submission(test_ids, binary_predictions, "sub.csv")