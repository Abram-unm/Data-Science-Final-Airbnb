from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.preprocessing import StandardScaler


def random_forest(XTrain, yTrain, XVal, criterion, max_depth):
    """
    Fits the data according to a support vector machine.
    Args:
    XTrain: numpy array of training data
    yTrain: numpy array of training class labels
    XVal: numpy array of validation data
    criterion: information gain criterion
    max_depth: max tree depth
    Returns:
    yVal: numpy array of predicted labels for validation
    """
    # Make Random Forest
    # Leave as 100 estimators
    model = RandomForestClassifier(criterion=criterion, max_depth=max_depth)
    
    # Fit model
    model.fit(XTrain, yTrain)

    # Make prediction
    yVal = model.predict(XVal)

    return yVal


def support_vector_machine(XTrain, yTrain, XVal, kernel, gamma, tol, max_iter):
    """
    Fits the data according to a support vector machine.
    Args:
    XTrain: numpy array of training data
    yTrain: numpy array of training class labels
    XVal: numpy array of validation data
    kernel: algorithm kernel type
    gamma: kernel coefficient hyperparameter for "rbf", "poly", and "sigmoid kernels
    tol: stopping error tolerance
    max_iter: max iterations for svm
    Returns:
    yVal: numpy array of predicted labels for validation
    """
    # Make SVM
    model = svm.SVC(kernel=kernel, gamma=gamma, tol=tol, max_iter=max_iter)

    # Scale first
    trainScaled, valScaled = scale_data(XTrain, XVal)

    # Fit model
    model.fit(trainScaled, yTrain)

    # Make predictions
    yVal = model.predict(valScaled)

    return yVal


def scale_data(XTrain, XVal):
    """
    Performs scaling of training and validation data.
    Args:
    XTrain: numpy array of training data
    XVal: numpy array of validation data
    Returns:
    XTrain_Scaled: numpy array of scaled training data
    XVal_Scaled: numpy array of scaledvalidation data
    """
    scaler = StandardScaler()
    
    # Apply standardization to data
    # Fit training data, NOT validation data
    XTrain_Scaled = scaler.fit_transform(XTrain)
    XVal_Scaled = scaler.transform(XVal)
    
    return XTrain_Scaled, XVal_Scaled