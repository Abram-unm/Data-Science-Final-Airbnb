import src.preprocessing as pre
import src.crossValidation as cv
import argparse
from hyperparameters import random_forest_hyperparameters, svm_hyperparameters


def main():

    parser = argparse.ArgumentParser(description='Random Forest Training Script')
    parser.add_argument('--data_path', type=str, default='dataset/train.csv',
                       help='Training data path (default: dataset/train.csv)')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory (default: results)')
    args = parser.parse_args()
    
    print("Beginning Random Forest")
    # Find hyperparameters for best Random Forest
    criterion, depth = random_forest_hyperparameters(args.data_path)
    print("Finished Random Forest")

    print("Beginning Support Vector Machine")
    # Find hyperparameters for best Support Vector Machine
    kernel, gamma, tolerance, iterations = svm_hyperparameters(args.data_path)
    print("Finished Support Vector Machine")

    print("Now using best parameters to train new models")

    data, labels = pre.load_training_data_all(args.data_path)
    print("\tData loaded")
    # PCA dimensionality reduction 
    use_pca = True
    n_components = 50
    if use_pca and data.shape[1] > n_components:
        print(f"  Applying PCA: {data.shape[1]} -> {n_components}")
        pca = PCA(n_components=n_components)
        data = pca.fit_transform(data)

    # Random Forest with best hyperparameters
    print("Final Random Forest\n")
    temp_data = data.copy()
    temp_labels = labels.copy()
    accuracies, pair, mean = cv.cross_validation_rf(temp_data, temp_labels, 5, criterion, depth)
    print("\tAccuracies: ")
    print("\t", accuracies)
    print("\tMean: ", mean)
    print("\n\n")

    # Support Vector Machine with best hyperparameters
    print("Final Support Vector Machine\n")
    temp_data = data.copy()
    temp_labels = labels.copy()
    accuracies, quad, mean = cv.cross_validation_svm(temp_data, temp_labels, 5, kernel, gamma, tolerance, iterations)
    print("\tAccuracies: ")
    print("\t", accuracies)
    print("\tMean: ", mean)
    print("\n\n")


if __name__ == "__main__":
    main()