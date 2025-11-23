import numpy as np

def load_data(file_path: str) -> np.ndarray:
    """
    Load the dataset from a given file path.
    Args:
        file_path: str, path to the data file
    Returns:
        data: numpy array of shape (n_samples, n_features)
    """
    return np.loadtxt(file_path)
