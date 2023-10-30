from helpers import *
import numpy as np
from implementations import *

#Load the data
datapath = "dataset/"
x_train, x_test, y_train, train_ids, test_ids = load_csv_data(datapath)

#Clean and Standardize the data
x_train_stripped, x_test_stripped = remove_empty_columns(x_train, x_test, 0.6)
x_train_standardized, x_test_standardized = standardize_data(x_train_stripped, x_test_stripped)

