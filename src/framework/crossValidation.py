import numpy as np
from . import other_models as om

# Seed randomness for future use
np.random.seed( 2 )



def shuffle_rows(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Shuffle the rows of the input matrix X.
    Args:
        X: data, numpy array of shape (n_samples, n_features)
        y: class labels, numpy array of shape (n_samples)
    Returns:
        X_shuffled: numpy array of shape (n_samples, n_features) with shuffled rows
        y_shuffled: numpy array of shape (n_samples,) with shuffled labels
    """
    # Shuffle rows according to seeded random
    shuffle = np.random.permutation( np.shape( X )[0] )
    return X[shuffle], y[shuffle]


def cross_validation_rf(X: np.ndarray, y: np.ndarray, k, criterion, max_depth):
    """
    Perform k-fold cross-validation on the dataset.

    Args:
        X: numpy array of shape (n_samples, n_features), data
        y: numpy array of shape (n_samples), class labels
        k: int, number of folds for cross-validation set at 5 for now
        criterion: information gain criterion
        max_depth: max tree depth
    Returns:
        ret: list of accuracies
        retCMD: list of model parameters for cross validation
        mean accuracy
    """
    # Shuffle data 
    shuffled_x, shuffled_y = shuffle_rows( X, y )
    # Setup splits
    split_size = np.shape( X )[0] // k
    split_one_x = shuffled_x[: split_size]
    split_one_y = shuffled_y[: split_size]
    split_two_x = shuffled_x[split_size : 2 * split_size]
    split_two_y = shuffled_y[split_size : 2 * split_size]
    split_three_x = shuffled_x[2 * split_size : 3 * split_size]
    split_three_y = shuffled_y[2 * split_size : 3 * split_size]
    split_four_x = shuffled_x[3 * split_size : 4 * split_size]
    split_four_y = shuffled_y[3 * split_size : 4 * split_size]
    split_five_x = shuffled_x[4 * split_size : 5 * split_size]
    split_five_y = shuffled_y[4 * split_size : 5 * split_size]
    splits_x = [split_one_x, split_two_x, split_three_x, split_four_x, split_five_x]
    splits_y = [split_one_y, split_two_y, split_three_y, split_four_y, split_five_y]
    ret = []
    retCMD = [criterion, max_depth]
    for i in range( k ):
        # Set up validation data
        split_val_x = splits_x[i]
        split_val_y = splits_y[i]
        # Set up training data
        splits_train_x = np.concatenate( splits_x[:i] + splits_x[i+1:] )
        splits_train_y = np.concatenate( splits_y[:i] + splits_y[i+1:] )
        # Below we want to call the code to generate class predictions and assign them to "predicts"
        predicts = om.random_forest(splits_train_x, splits_train_y, split_val_x, criterion, max_depth)
        # Evaluate accuracy of predictions
        correct = 0
        for j in range( np.shape( predicts )[0] ):
            if predicts[j] == split_val_y[j]:
                correct += 1
        accuracy = correct * 100 / np.shape( predicts )[0]
        ret.append( ( accuracy ) )
    # Return list with accuracy of each fold
    return ret, retCMD, np.mean(ret)


def cross_validation_gnb(X: np.ndarray, y: np.ndarray, k: int = 5):
    """
    Perform k-fold cross-validation on the dataset.

    Args:
        X: numpy array of shape (n_samples, n_features), data
        y: numpy array of shape (n_samples), class labels
        k: int, number of folds for cross-validation set at 5 for now
    Returns:
        ret: list of accuracies
        mean accuracy
    """
    # Shuffle data 
    shuffled_x, shuffled_y = shuffle_rows( X, y )
    # Setup splits
    split_size = np.shape( X )[0] // k
    split_one_x = shuffled_x[: split_size]
    split_one_y = shuffled_y[: split_size]
    split_two_x = shuffled_x[split_size : 2 * split_size]
    split_two_y = shuffled_y[split_size : 2 * split_size]
    split_three_x = shuffled_x[2 * split_size : 3 * split_size]
    split_three_y = shuffled_y[2 * split_size : 3 * split_size]
    split_four_x = shuffled_x[3 * split_size : 4 * split_size]
    split_four_y = shuffled_y[3 * split_size : 4 * split_size]
    split_five_x = shuffled_x[4 * split_size : 5 * split_size]
    split_five_y = shuffled_y[4 * split_size : 5 * split_size]
    splits_x = [split_one_x, split_two_x, split_three_x, split_four_x, split_five_x]
    splits_y = [split_one_y, split_two_y, split_three_y, split_four_y, split_five_y]
    ret = []
    for i in range( k ):
        # Set up validation data
        split_val_x = splits_x[i]
        split_val_y = splits_y[i]
        # Set up training data
        splits_train_x = np.concatenate( splits_x[:i] + splits_x[i+1:] )
        splits_train_y = np.concatenate( splits_y[:i] + splits_y[i+1:] )
        # Below we want to call the code to generate class predictions and assign them to "predicts"
        predicts = om.gaussian_naive_bayes(splits_train_x, splits_train_y, split_val_x)
        # Evaluate accuracy of predictions
        correct = 0
        for j in range( np.shape( predicts )[0] ):
            if predicts[j] == split_val_y[j]:
                correct += 1
        accuracy = correct * 100 / np.shape( predicts )[0]
        ret.append( ( accuracy ) )
    # Return list with accuracy of each fold
    return ret, np.mean(ret)


def cross_validation_svm(X: np.ndarray, y: np.ndarray, k, kernel, gamma, tol, max_iters):
    """
    Perform k-fold cross-validation on the dataset.

    Args:
        X: numpy array of shape (n_samples, n_features), data
        y: numpy array of shape (n_samples), class labels
        k: int, number of folds for cross-validation set at 5 for now
        kernel: algorithm kernel type
        gamma: kernel coefficient hyperparameter for "rbf", "poly", and "sigmoid kernels
        tol: stopping error tolerance
        max_iter: max iterations for svm
    Returns:
        ret: list of accuracies
        retCMD: list of model parameters for cross validation
        mean accuracy
    """
    # Shuffle data 
    shuffled_x, shuffled_y = shuffle_rows( X, y )
    # Setup splits
    split_size = np.shape( X )[0] // k
    split_one_x = shuffled_x[: split_size]
    split_one_y = shuffled_y[: split_size]
    split_two_x = shuffled_x[split_size : 2 * split_size]
    split_two_y = shuffled_y[split_size : 2 * split_size]
    split_three_x = shuffled_x[2 * split_size : 3 * split_size]
    split_three_y = shuffled_y[2 * split_size : 3 * split_size]
    split_four_x = shuffled_x[3 * split_size : 4 * split_size]
    split_four_y = shuffled_y[3 * split_size : 4 * split_size]
    split_five_x = shuffled_x[4 * split_size : 5 * split_size]
    split_five_y = shuffled_y[4 * split_size : 5 * split_size]
    splits_x = [split_one_x, split_two_x, split_three_x, split_four_x, split_five_x]
    splits_y = [split_one_y, split_two_y, split_three_y, split_four_y, split_five_y]
    ret = []
    retCMD = [kernel, gamma, tol, max_iters]
    for i in range( k ):
        # Set up validation data
        split_val_x = splits_x[i]
        split_val_y = splits_y[i]
        # Set up training data
        splits_train_x = np.concatenate( splits_x[:i] + splits_x[i+1:] )
        splits_train_y = np.concatenate( splits_y[:i] + splits_y[i+1:] )
        # Below we want to call the code to generate class predictions and assign them to "predicts"
        predicts = om.support_vector_machine(splits_train_x, splits_train_y, split_val_x, kernel, gamma, tol, max_iters)
        # Evaluate accuracy of predictions
        correct = 0
        for j in range( np.shape( predicts )[0] ):
            if predicts[j] == split_val_y[j]:
                correct += 1
        accuracy = correct * 100 / np.shape( predicts )[0]
        ret.append( ( accuracy ) )
    # Return list with accuracy of each fold
    return ret, retCMD, np.mean(ret)


def cross_validation_gbm(X: np.ndarray, y: np.ndarray, k, lr):
    """
    Perform k-fold cross-validation on the dataset.

    Args:
        X: numpy array of shape (n_samples, n_features), data
        y: numpy array of shape (n_samples), class labels
        k: int, number of folds for cross-validation set at 5 for now
        learning_rate: specific learning rate for GBM classifier
    Returns:
        ret: list of accuracies
        retCMD: learning rate for GBM
        mean accuracy
    """
    # Shuffle data 
    shuffled_x, shuffled_y = shuffle_rows( X, y )
    # Setup splits
    split_size = np.shape( X )[0] // k
    split_one_x = shuffled_x[: split_size]
    split_one_y = shuffled_y[: split_size]
    split_two_x = shuffled_x[split_size : 2 * split_size]
    split_two_y = shuffled_y[split_size : 2 * split_size]
    split_three_x = shuffled_x[2 * split_size : 3 * split_size]
    split_three_y = shuffled_y[2 * split_size : 3 * split_size]
    split_four_x = shuffled_x[3 * split_size : 4 * split_size]
    split_four_y = shuffled_y[3 * split_size : 4 * split_size]
    split_five_x = shuffled_x[4 * split_size : 5 * split_size]
    split_five_y = shuffled_y[4 * split_size : 5 * split_size]
    splits_x = [split_one_x, split_two_x, split_three_x, split_four_x, split_five_x]
    splits_y = [split_one_y, split_two_y, split_three_y, split_four_y, split_five_y]
    ret = []
    retCMD = lr
    for i in range( k ):
        # Set up validation data
        split_val_x = splits_x[i]
        split_val_y = splits_y[i]
        # Set up training data
        splits_train_x = np.concatenate( splits_x[:i] + splits_x[i+1:] )
        splits_train_y = np.concatenate( splits_y[:i] + splits_y[i+1:] )
        # Below we want to call the code to generate class predictions and assign them to "predicts"
        predicts = om.gradient_boosting_machine(splits_train_x, splits_train_y, split_val_x, lr)
        # Evaluate accuracy of predictions
        correct = 0
        for j in range( np.shape( predicts )[0] ):
            if predicts[j] == split_val_y[j]:
                correct += 1
        accuracy = correct * 100 / np.shape( predicts )[0]
        ret.append( ( accuracy ) )
    # Return list with accuracy of each fold
    return ret, retCMD, np.mean(ret)
