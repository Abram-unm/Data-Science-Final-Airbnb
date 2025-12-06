from hyperparameters import random_forest_hyperparameters
from models import random_forest
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from preprocessing import load_data_rf
import os


def main():
    print("Beginning Random Forest")
    # Find hyperparameters for best Random Forest
    train_data, train_labels = load_data_rf("../prepared/train_preprocessed.csv")
    test_data, test_labels = load_data_rf("../prepared/test_preprocessed.csv")
    criterion, depth = random_forest_hyperparameters(train_data, train_labels)
    print("Finished Random Forest")

    print("Now using best parameters to train new models")

    print("\tData loaded")
    # PCA dimensionality reduction 
    use_pca = True
    n_components = 50
    if use_pca and train_data.shape[1] > n_components:
        print(f"  Applying PCA: {train_data.shape[1]} -> {n_components}")
        pca = PCA(n_components=n_components)
        train_data = pca.fit_transform(train_data)
        print(f"  Applying PCA: {test_data.shape[1]} -> {n_components}")
        test_data = pca.transform(test_data)

    # Random Forest with best hyperparameters
    print("Final Random Forest\n")
    preds = random_forest(train_data, train_labels, test_data, criterion, depth)
    # Gather Confusion matrix
    conf = confusion_matrix(test_labels, preds)
    pict = ConfusionMatrixDisplay(confusion_matrix=conf)
    title_string = f"Random Forest: IG Criterion={criterion} Max Depth={depth}"
    #pict.title(title_string)
    pict.plot
    # Save the plot
    plot_pretrain = f"random_forest_plot_info_gain_criterion={criterion}_max_depth={depth}.png"
    plt.savefig(plot_pretrain)
    plt.close()


if __name__ == "__main__":
    main()