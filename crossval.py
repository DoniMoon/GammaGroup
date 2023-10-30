from implementations import *


def cross_validation_thresh_log_reg(y, x, k_fold, threshs, seed=1):
    """Cross validation on the threshold hyperparameter in  logistic regression model 
    
    Args:
        y:          shape=(N,)
        x:          shape=(N,D). D is the number of features
        k_fold:     scalar, number of folds
        threshs:    array of potential threshholders
        seed:       seed for numpy.random

    Returns:
       best_thresh:     threshold corresponding to lowest test_accuracy 
       best_train_acc:  train accuracy corresponding to best threshhold
       best_test_acc:   test accuracy corresponding to best threshhold

    """
    k_indices = build_k_indices(y, k_fold, seed)

    accs_test = []
    accs_train = []
    for i, thresh in enumerate(threshs):
        acc_test = 0
        acc_train = 0
        fp = 0
        fn = 0
        for k in range(k_fold):
            # split data
            te_indice = k_indices[k]
            tr_indice = np.delete(k_indices, k, axis=0)
            tr_indice = tr_indice.reshape(-1)
            x_test = x[te_indice]
            y_test = y[te_indice]
            x_train = x[tr_indice]
            y_train = y[tr_indice]

            x_train, mean, std = standardize(x_train)
            x_test = (x_test - mean) / std

            # add bias
            x_train = np.insert(x_train, 0, np.ones(x_train.shape[0]), axis=1)
            x_test = np.insert(x_test, 0, np.ones(x_test.shape[0]), axis=1)
            # fit data
            initial_w = np.zeros(x_train.shape[1],)
            w, loss = logistic_regression(
                y_train, x_train, initial_w, max_iters=2000, gamma=0.1
            )

            # calculate accuracy , false positives and false negatives
            f = np.vectorize(lambda x: 0 if x < thresh else 1)
            pred_train = f(sigmoid(x_train.dot(w)))
            pred_test = f(sigmoid(x_test.dot(w)))
            acc_train += accuracy(y_train, pred_train)
            acc_test += accuracy(y_test, pred_test)
            fp += false_positives(y_test, pred_test, neg=0)
            fn += false_negatives(y_test, pred_test, neg=0)

        # average accuracy , false positives and false negatives accross k folds
        acc_test /= k_fold
        acc_train /= k_fold
        fp /= k_fold
        fn /= k_fold

        accs_test.append(acc_test)
        accs_train.append(acc_train)
        print(
            "for thresh {t} average train accuracy {at} and average test acc {ate} \n false positives {fp} false negative {fn}".format(
                t=thresh, at=acc_train, ate=acc_test, fp=fp, fn=fn
            )
        )

    index_max_acc = np.argmax(
        accs_test
    )  # return the index correspond to the best accuracy
    best_test_acc = accs_test[index_max_acc]
    best_train_acc = accs_train[index_max_acc]
    best_thresh = threshs[index_max_acc]

    print(
        "cross validation on thresh , best acc_test {ate} , train_acc {at} corresponding to thresh {t} ".format(
            ate=best_test_acc, at=best_train_acc, t=best_thresh
        )
    )

    return best_thresh, best_train_acc, best_test_acc


def cross_validation_thresh_least_sq(y, x, k_fold, threshs, seed=1):
    """Cross validation on the threshold hyperparameter in  least squares  model 
    
    Args:
        y:          shape=(N,)
        x:          shape=(N,D). D is the number of features
        k_fold:     scalar, number of folds
        threshs:    array of potential threshholders
        seed:       seed for numpy.random

    Returns:
       best_thresh:     threshold corresponding to lowest test_accuracy 
       best_train_acc:  train accuracy corresponding to best threshhold
       best_test_acc:   test accuracy corresponding to best threshhold

    """
    k_indices = build_k_indices(y, k_fold, seed)

    accs_test = []
    accs_train = []
    for i, thresh in enumerate(threshs):
        acc_test = 0
        acc_train = 0
        fp = 0
        fn = 0
        for k in range(k_fold):
            te_indice = k_indices[k]
            tr_indice = np.delete(k_indices, k, axis=0)
            tr_indice = tr_indice.reshape(-1)
            x_test = x[te_indice]
            y_test = y[te_indice]
            x_train = x[tr_indice]
            y_train = y[tr_indice]

            x_train, mean, std = standardize(x_train)
            x_test = (x_test - mean) / std

            # add bias
            x_train = np.insert(x_train, 0, np.ones(x_train.shape[0]), axis=1)
            x_test = np.insert(x_test, 0, np.ones(x_test.shape[0]), axis=1)

            w, loss = least_squares(y_train, x_train)

            f = np.vectorize(lambda x: -1 if x < thresh else 1)
            pred_train = f(x_train.dot(w))
            pred_test = f(x_test.dot(w))
            acc_train += accuracy(y_train, pred_train)
            acc_test += accuracy(y_test, pred_test)
            fp += false_positives(y_test, pred_test)
            fn += false_negatives(y_test, pred_test)

        acc_test /= k_fold
        acc_train /= k_fold
        fp /= k_fold
        fn /= k_fold

        accs_test.append(acc_test)
        accs_train.append(acc_train)
        print(
            "for thresh {t} average train accuracy {at} and average test acc {ate}\n false positives {fp} false negatives {fn}".format(
                t=thresh, at=acc_train, ate=acc_test, fp=fp, fn=fn
            )
        )

    index_max_acc = np.argmax(accs_test)
    best_test_acc = accs_test[index_max_acc]
    best_train_acc = accs_train[index_max_acc]
    best_thresh = threshs[index_max_acc]

    print(
        "cross validation on thresh , best acc_test {ate} , train_acc {at} corresponding to thresh {t} ".format(
            ate=best_test_acc, at=best_train_acc, t=best_thresh
        )
    )

    return best_thresh, best_train_acc, best_test_acc


def cross_validation_degree_lambda_log_reg(y, x, k_fold, lambdas, seed=1):
    """Cross validation on the lambda hyperparameter in  logistic regression model 
    
    Args:
        y:          shape=(N,)
        x:          shape=(N,D). D is the number of features
        k_fold:     scalar, number of folds
        lambdas:    array of potential lambdas
        seed:       seed for numpy.random

    Returns:
       best_lambda:     lambda corresponding to lowest test_accuracy 
       best_train_acc:  train accuracy corresponding to best lambda
       best_test_acc:   test accuracy corresponding to best lambda

    """
    k_indices = build_k_indices(y, k_fold, seed)

    accs_test = []
    accs_train = []
    for i, lambda_ in enumerate(lambdas):
        acc_test = 0
        acc_train = 0
        fp = 0
        fn = 0
        for k in range(k_fold):
            te_indice = k_indices[k]
            tr_indice = np.delete(k_indices, k, axis=0)
            tr_indice = tr_indice.reshape(-1)
            x_test = x[te_indice]
            y_test = y[te_indice]
            x_train = x[tr_indice]
            y_train = y[tr_indice]

            x_train, mean, std = standardize(x_train)
            x_test = (x_test - mean) / std

            x_train = np.insert(x_train, 0, np.ones(x_train.shape[0]), axis=1)
            x_test = np.insert(x_test, 0, np.ones(x_test.shape[0]), axis=1)

            initial_w = np.random.rand(x_train.shape[1],)
            w, loss = reg_logistic_regression(
                y_train, x_train, lambda_, initial_w, max_iters=2000, gamma=0.1
            )
            print(loss)
            f = np.vectorize(lambda x: 0 if x < 0.5 else 1)
            pred_train = f(sigmoid(x_train.dot(w)))
            pred_test = f(sigmoid(x_test.dot(w)))
            acc_train += accuracy(y_train, pred_train)
            acc_test += accuracy(y_test, pred_test)
            fp += false_positives(y_test, pred_test, neg=0)
            fn += false_negatives(y_test, pred_test, neg=0)

        acc_test /= k_fold
        acc_train /= k_fold
        fp /= k_fold
        fn /= k_fold

        accs_test.append(acc_test)
        accs_train.append(acc_train)
        print(
            "for lambda {t} average train accuracy {at} and average test acc {ate}\n false postives {fp} false negatives {fn}".format(
                t=lambda_, at=acc_train, ate=acc_test, fp=fp, fn=fn
            )
        )

    index_max_acc = np.argmax(accs_test)
    best_test_acc = accs_test[index_max_acc]
    best_train_acc = accs_train[index_max_acc]
    best_lambda = lambdas[index_max_acc]

    print(
        "cross validation on lambda , best acc_test {ate} , train_acc {at} corresponding to lambda {t} ".format(
            ate=best_test_acc, at=best_train_acc, t=best_lambda
        )
    )

    return best_lambda, best_train_acc, best_test_acc


def cross_validation_degree_lambda_ridge(y, x, k_fold, pairs, seed=1):
    """return the loss of ridge regression for a fold corresponding to k_indices

    Args:
        y:          shape=(N,)
        x:          shape=(N,)
        k_indices:  2D array returned by build_k_indices()
        k:          scalar, the k-th fold (N.B.: not to confused with k_fold which is the fold nums)
        lambda_:    scalar, cf. ridge_regression()
        degree:     scalar, cf. build_poly()

    Returns:
        train and test root mean square errors rmse = sqrt(2 mse)

    """
    k_indices = build_k_indices(y, k_fold, seed)

    accs_test = []
    accs_train = []
    for degree, lambda_ in pairs:
        x_poly = build_poly(x, degree)
        acc_test = 0
        acc_train = 0
        for k in range(k_fold):
            te_indice = k_indices[k]
            tr_indice = np.delete(k_indices, k, axis=0)
            tr_indice = tr_indice.reshape(-1)
            x_test = x_poly[te_indice]
            y_test = y[te_indice]
            x_train = x_poly[tr_indice]
            y_train = y[tr_indice]

            x_train, mean, std = standardize(x_train)
            x_test = (x_test - mean) / std

            # add bias
            x_train = np.insert(x_train, 0, np.ones(x_train.shape[0]), axis=1)
            x_test = np.insert(x_test, 0, np.ones(x_test.shape[0]), axis=1)
            w, loss = ridge_regression(y_train, x_train, lambda_)

            f = np.vectorize(lambda x: -1 if x <= 0 else 1)
            pred_train = f(x_train.dot(w))
            pred_test = f(x_test.dot(w))
            acc_train += accuracy(y_train, pred_train)
            acc_test += accuracy(y_test, pred_test)

        acc_test /= k_fold
        acc_train /= k_fold

        accs_test.append(acc_test)
        accs_train.append(acc_train)
        print(
            "for degree {d}and lambda {t} average train accuracy {at} and average test acc {ate}".format(
                d=degree, t=lambda_, at=acc_train, ate=acc_test
            )
        )

    index_max_acc = np.argmax(accs_test)
    best_test_acc = accs_test[index_max_acc]
    best_train_acc = accs_train[index_max_acc]
    best_degree, best_lambda = pairs[index_max_acc]

    print(
        "cross validation on (degree, lambda) , best acc_test {ate} , train_acc {at} corresponding to degree  {d} and lambda {l} ".format(
            ate=best_test_acc, at=best_train_acc, d=best_degree, l=best_lambda
        )
    )

    return best_degree, best_lambda, best_train_acc, best_test_acc


def cross_validation_degree_least_sq(y, x, k_fold, degrees, seed=1):
    """Cross validation on the polynomial expansion degree hyperparameter in  logistic regression model 
    
    Args:
        y:          shape=(N,)
        x:          shape=(N,D). D is the number of features
        k_fold:     scalar, number of folds
        threshs:    array of potential threshholders
        seed:       seed for numpy.random

    Returns:
       best_thresh:     threshold corresponding to lowest test_accuracy 
       best_train_acc:  train accuracy corresponding to best threshhold
       best_test_acc:   test accuracy corresponding to best threshhold

    """
    k_indices = build_k_indices(y, k_fold, seed)

    accs_test = []
    accs_train = []
    for i, degree in enumerate(degrees):

        x_poly = build_poly(x, degree)

        acc_test = 0
        acc_train = 0
        fp = 0
        fn = 0
        for k in range(k_fold):
            te_indice = k_indices[k]
            tr_indice = np.delete(k_indices, k, axis=0)
            tr_indice = tr_indice.reshape(-1)
            x_test = x_poly[te_indice]
            y_test = y[te_indice]
            x_train = x_poly[tr_indice]
            y_train = y[tr_indice]

            x_train, mean, std = standardize(x_train)
            x_test = (x_test - mean) / std

            # add bias
            x_train = np.insert(x_train, 0, np.ones(x_train.shape[0]), axis=1)
            x_test = np.insert(x_test, 0, np.ones(x_test.shape[0]), axis=1)

            w, loss = least_squares(y_train, x_train)
            print(loss)
            f = np.vectorize(lambda x: -1 if x < 0 else 1)
            pred_train = f(x_train.dot(w))
            pred_test = f(x_test.dot(w))
            acc_train += accuracy(y_train, pred_train)
            acc_test += accuracy(y_test, pred_test)
            fp += false_positives(y_test, pred_test)
            fn += false_negatives(y_test, pred_test)

        acc_test /= k_fold
        acc_train /= k_fold
        fp /= k_fold
        fn /= k_fold

        accs_test.append(acc_test)
        accs_train.append(acc_train)
        # print('for degree {t} average train accuracy {at} and average test acc {ate}\n false positives {fp} false negatives {fn}'.format(t=degree,at=acc_train,ate=acc_test,fp=fp, fn=fn))

    index_max_acc = np.argmax(accs_test)
    best_test_acc = accs_test[index_max_acc]
    best_train_acc = accs_train[index_max_acc]
    best_degree = degrees[index_max_acc]

    print(
        "cross validation on degree , best acc_test {ate} , train_acc {at} corresponding to degree {t} ".format(
            ate=best_test_acc, at=best_train_acc, t=best_degree
        )
    )

    return best_degree, best_train_acc, best_test_acc


def cross_validation_degree_log(y, x, k_fold, degrees, seed=1):
    """Cross validation on the polynomial expansion degree hyperparameter in  logistic regression model 
    
    Args:
        y:          shape=(N,)
        x:          shape=(N,D). D is the number of features
        k_fold:     scalar, number of folds
        threshs:    array of potential threshholders
        seed:       seed for numpy.random

    Returns:
       best_thresh:     threshold corresponding to lowest test_accuracy 
       best_train_acc:  train accuracy corresponding to best threshhold
       best_test_acc:   test accuracy corresponding to best threshhold

    """

    k_indices = build_k_indices(y, k_fold, seed)

    accs_test = []
    accs_train = []
    for i, degree in enumerate(degrees):
        x_poly = build_poly(x, degree)

        acc_test = 0
        acc_train = 0
        fp = 0
        fn = 0
        for k in range(k_fold):
            te_indice = k_indices[k]
            tr_indice = np.delete(k_indices, k, axis=0)
            tr_indice = tr_indice.reshape(-1)
            x_test = x_poly[te_indice]
            y_test = y[te_indice]
            x_train = x_poly[tr_indice]
            y_train = y[tr_indice]

            x_train, mean, std = standardize(x_train)
            x_test = (x_test - mean) / std

            # add bias
            x_train = np.insert(x_train, 0, np.ones(x_train.shape[0]), axis=1)
            x_test = np.insert(x_test, 0, np.ones(x_test.shape[0]), axis=1)

            initial_w = np.random.rand(x_train.shape[1],)
            w, loss = logistic_regression(
                y_train, x_train, initial_w, max_iters=3000, gamma=0.1
            )
            print(loss)
            f = np.vectorize(lambda x: 0 if x < 0.5 else 1)
            pred_train = f(sigmoid(x_train.dot(w)))
            pred_test = f(sigmoid(x_test.dot(w)))
            acc_train += accuracy(y_train, pred_train)
            acc_test += accuracy(y_test, pred_test)
            fp += false_positives(y_test, pred_test, neg=0)
            fn += false_negatives(y_test, pred_test, neg=0)

        acc_test /= k_fold
        acc_train /= k_fold
        fp /= k_fold
        fn /= k_fold

        accs_test.append(acc_test)
        accs_train.append(acc_train)
        print(
            "for degree {t} average train accuracy {at} and average test acc {ate}\n false postives {fp} false negatives {fn}".format(
                t=degree, at=acc_train, ate=acc_test, fp=fp, fn=fn
            )
        )

    index_max_acc = np.argmax(accs_test)
    best_test_acc = accs_test[index_max_acc]
    best_train_acc = accs_train[index_max_acc]
    best_degree = degrees[index_max_acc]

    print(
        "cross validation on degree , best acc_test {ate} , train_acc {at} corresponding to degree {t} ".format(
            ate=best_test_acc, at=best_train_acc, t=best_degree
        )
    )

    return best_degree, best_train_acc, best_test_acc
