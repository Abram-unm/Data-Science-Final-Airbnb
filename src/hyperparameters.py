from sklearn.decomposition import PCA
import numpy as np
import crossValidation as cv
import time

def random_forest_hyperparameters(data, labels):
    """
    Finds the best hyperparameters for Random Forest.
    Args:
        data: numpy array with training data
        labels: numpy labels with training data
    Returns:
        Best information gain criterion for random forest
        Best max depth for random forest
    """
    criterion_list = ["gini", "entropy", "log_loss"]
    depth_list = [5, 10, 15, 20]

    print("\tData loaded")
    # PCA dimensionality reduction 
    use_pca = True
    n_components = 50
    if use_pca and data.shape[1] > n_components:
        print(f"  Applying PCA: {data.shape[1]} -> {n_components}")
        pca = PCA(n_components=n_components)
        data = pca.fit_transform(data)

    print("All data with rf:\n")
    pairs = []
    means = []
    accs = []
    times = []
    for i in range(len(criterion_list)):
        for j in range(len(depth_list)):
            start = time.time( )
            # Do not reload all data
            temp_data = data.copy()
            temp_labels = labels.copy()
            # Run cross validation with hyperparameters
            accuracies, pair, mean = cv.cross_validation_rf(temp_data, temp_labels, 5, criterion_list[i], depth_list[j])
            end = time.time( )
            print("\tAccuracies: ")
            print("\t", accuracies)
            print("\tMean: ", mean)
            print("\tCombination: ", pair[0], pair[1])
            print("\tExecution time: ", end - start)
            print("\n")
            pairs.append(pair)
            means.append(mean)
            accs.append(accuracies)
            times.append(end - start)
    # Find best mean accuracy
    index = np.argmax(means)
    max_mean = np.max(means)
    print("Maximum mean for all data: ", max_mean)
    print("Best hyperparameter pair for all data: ", pairs[index][0], pairs[index][1])
    print("Best accuracies: ", accs[index])
    print("Best execution time: ", times[index])
    print("\n\n")
    return pairs[index][0], pairs[index][1]


def svm_hyperparameters(data, labels):
    """
    Finds the best hyperparameters for Support Vector Machine.
    Args:
        data: numpy array with training data
        labels: numpy labels with training data
    Returns:
        Best kernel type for svm
        Best kernel coefficient method for svm
        Best error tolerance for svm
        Best max iterations for svm
    """
    kernel_list = ["linear", "poly", "rbf", "sigmoid"]
    gamma_list = ["scale", "auto"]
    tolerance_list = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
    iterations_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, -1]

    print("\tData loaded")
    # PCA dimensionality reduction 
    use_pca = True
    n_components = 50
    if use_pca and data.shape[1] > n_components:
        print(f"  Applying PCA: {data.shape[1]} -> {n_components}")
        pca = PCA(n_components=n_components)
        data = pca.fit_transform(data)

    print("All data with svm:\n")
    quads = []
    means = []
    accs = []
    times = []
    for i in range(len(kernel_list)):
        for j in range(len(gamma_list)):
            for k in range(len(tolerance_list)):
                for l in range(len(iterations_list)):
                    start = time.time( )
                    # Do not reload all data
                    temp_data = data.copy()
                    temp_labels = labels.copy()
                    # Run cross validation with hyperparameters
                    accuracies, quad, mean = cv.cross_validation_svm(temp_data, temp_labels, 5, kernel_list[i], gamma_list[j], tolerance_list[k], iterations_list[l])
                    end = time.time( )
                    print("\tAccuracies: ")
                    print("\t", accuracies)
                    print("\tMean: ", mean)
                    print("\tCombination: ", quad[0], ",", quad[1], ",", quad[2], ",", quad[3])
                    print("\tExecution time: ", end - start)
                    print("\n")
                    quads.append(quad)
                    means.append(mean)
                    accs.append(accuracies)
                    times.append(end - start)
    # Find best mean accuracy
    index = np.argmax(means)
    max_mean = np.max(means)
    print("Maximum mean for all data: ", max_mean)
    print("Best hyperparameter combination for all data: ", quads[index][0], ",", quads[index][1], ",", quads[index][2], ",", quads[index][3])
    print("Best accuracies: ", accs[index])
    print("Best execution time: ", times[index])
    print("\n\n")
    return quads[index][0], quads[index][1], quads[index][2], quads[index][3]
