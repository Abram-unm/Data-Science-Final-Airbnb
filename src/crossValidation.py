import numpy as np

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

# Update this to have the parameters that are desired
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
    ret = []
    # Update this to reflect actual parameters
    retCMD = [criterion, max_depth]
    for i in range( k ):
        # Set up validation data
        split_val_x = shuffled_x[(i*split_size):((i+1)*split_size)]
        split_val_y = shuffled_y[(i*split_size):((i+1)*split_size)]
        # Set up training data
        splits_train_x = np.concatenate( shuffled_x[:(i*split_size)] + shuffled_x[((i+1)*split_size):] )
        splits_train_y = np.concatenate( shuffled_y[:(i*split_size)] + shuffled_y[((i+1)*split_size):] )
        # Below we want to call the code to generate class predictions and assign them to "predicts"
        #predicts = om.random_forest(splits_train_x, splits_train_y, split_val_x, criterion, max_depth)
        # Evaluate accuracy of predictions
        correct = 0
        for j in range( np.shape( predicts )[0] ):
            if predicts[j] == split_val_y[j]:
                correct += 1
        accuracy = correct * 100 / np.shape( predicts )[0]
        ret.append( ( accuracy ) )
    # Return list with accuracy of each fold
    return ret, retCMD, np.mean(ret)
