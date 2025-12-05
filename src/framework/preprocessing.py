import os
import numpy as np
import librosa


GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 
          'jazz', 'metal', 'pop', 'reggae', 'rock']


def load_training_data_stft(folderpath):
    """
    Load the data from different directories, collect per-file Short-Term Fourier Transform average amplitude over time slices
    Args:
        folderpath: path to data directories
    Returns:
        numpy array of data
    """
    data = [] 
    for genre in GENRES:
        genre_folder = os.path.join(folderpath, genre)
        if not os.path.exists(genre_folder):
            continue
        for file in os.listdir(genre_folder):
            if not file.endswith('.au'):
                continue
            try:
                y, sr = librosa.core.load(os.path.join(genre_folder, file), sr=None)
                stft = librosa.stft(y=y)
                # Get amplitudes
                data_amp = np.abs(stft)
                # Average amplitude over time frames
                data.append(np.mean(data_amp, axis=1))
            except Exception as e:
                print(f"Error processing file {file}: {e}")
                continue
    return np.array(data)


def load_training_data_mfcc(folderpath):
    """
    Load the data from different directories, collect per-file Mel Frequency Cepstral Coefficients average over coefficients
    Args:
        folderpath: path to data directories
    Returns:
        numpy array of data
    """
    # 13 is good number of coefficients
    data = []
    for genre in GENRES:
        genre_folder = os.path.join(folderpath, genre)
        if not os.path.exists(genre_folder):
            continue
        for file in os.listdir(genre_folder):
            if not file.endswith('.au'):
                continue
            try:
                y, sr = librosa.core.load(os.path.join(genre_folder, file), sr=None)
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                # Average mfccs over time
                data.append(np.mean(mfcc, axis=1))
            except Exception as e:
                print(f"Error processing file {file}: {e}")
                continue
    return np.array(data)


def load_training_data_cf(folderpath):
    """
    Load the data from different directories, collect per-file Chroma feature average over chroma bin
    Args:
        folderpath: path to data directories
    Returns:
        numpy array of data
    """
    data = []
    for genre in GENRES:
        genre_folder = os.path.join(folderpath, genre)
        if not os.path.exists(genre_folder):
            continue
        for file in os.listdir(genre_folder):
            if not file.endswith('.au'):
                continue
            try:
                y, sr = librosa.core.load(os.path.join(genre_folder, file), sr=None)
                cf = librosa.feature.chroma_stft(y=y, sr=sr)
                # Average each chroma bin over time
                data.append(np.mean(cf, axis=1))
            except Exception as e:
                print(f"Error processing file {file}: {e}")
                continue
    return np.array(data)


def load_training_data_sc(folderpath):
    """
    Load the data from different directories, collect per-file spectral contrast average over time
    Args:
        folderpath: path to data directories
    Returns:
        numpy array of data
    """
    data = []
    for genre in GENRES:
        genre_folder = os.path.join(folderpath, genre)
        if not os.path.exists(genre_folder):
            continue
        for file in os.listdir(genre_folder):
            if not file.endswith('.au'):
                continue
            try:
                y, sr = librosa.core.load(os.path.join(genre_folder, file), sr=None)
                sc = librosa.feature.spectral_contrast(y=y, sr=sr)
                # Average over time
                data.append(np.mean(sc, axis=1))
            except Exception as e:
                print(f"Error processing file {file}: {e}")
                continue
    return np.array(data)


def load_training_data_zcr(folderpath):
    """
    Load the data from different directories, collect per-file zero-crossing rate over time
    Args:
        folderpath: path to data directories
    Returns:
        numpy array of data
    """
    data = []
    for genre in GENRES:
        genre_folder = os.path.join(folderpath, genre)
        if not os.path.exists(genre_folder):
            continue
        for file in os.listdir(genre_folder):
            if not file.endswith('.au'):
                continue
            try:
                y, sr = librosa.core.load(os.path.join(genre_folder, file), sr=None)
                zcr = librosa.feature.zero_crossing_rate(y=y)
                # Average and std over time
                zcr_mean = np.mean(zcr)
                zcr_std = np.std(zcr)
                data.append(np.array([zcr_mean, zcr_std]))
            except Exception as e:
                print(f"Error processing file {file}: {e}")
                continue
    return np.array(data)


def load_training_data_energy(folderpath):
    """
    Load the data from different directories, collect per-file energy over time
    Args:
        folderpath: path to data directories
    Returns:
        numpy array of data
    """
    data = []
    for genre in GENRES:
        genre_folder = os.path.join(folderpath, genre)
        if not os.path.exists(genre_folder):
            continue
        for file in os.listdir(genre_folder):
            if not file.endswith('.au'):
                continue
            try:
                y, sr = librosa.load(os.path.join(genre_folder, file), None)
                energy = librosa.feature.rms(y=y)
                # Average and std over time
                energy_mean = np.mean(energy)
                energy_std = np.std(energy)
                data.append(np.array([energy_mean, energy_std]))
            except Exception as e:
                print(f"Error processing file {file}: {e}")
                continue
    return np.array(data)


def load_training_data_all_features(folderpath):
    """
    Extract all features with a single pass per file to keep lengths aligned.
    Return X, y, genre_to_id.
    """
    print("Extracting all features ...........")
    genre_to_id = {genre: idx for idx, genre in enumerate(GENRES)}

    X_rows = []
    y_rows = []

    for genre in GENRES:
        genre_folder = os.path.join(folderpath, genre)
        if not os.path.exists(genre_folder):
            continue
        for file in os.listdir(genre_folder):
            if not file.endswith('.au'):
                continue
            file_path = os.path.join(genre_folder, file)
            try:
                y_audio, sr = librosa.core.load(file_path, sr=None)
                if y_audio is None or len(y_audio) == 0:
                    print(f"{file}: empty audio, skipped")
                    continue

                # STFT
                stft = librosa.stft(y=y_audio)
                stft_amp = np.abs(stft)
                stft_mean = np.mean(stft_amp, axis=1)

                # MFCC
                mfcc = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=13)
                mfcc_mean = np.mean(mfcc, axis=1)

                # Chroma
                chroma = librosa.feature.chroma_stft(y=y_audio, sr=sr)
                chroma_mean = np.mean(chroma, axis=1)

                # Spectral Contrast
                try:
                    spectral_contrast = librosa.feature.spectral_contrast(y=y_audio, sr=sr)
                    spectral_contrast_mean = np.mean(spectral_contrast, axis=1)
                except Exception as e:
                    print(f" {file}: {e}")
                    spectral_contrast_mean = np.zeros(7)

                # ZCR
                zcr = librosa.feature.zero_crossing_rate(y=y_audio)
                zcr_mean = np.mean(zcr)
                zcr_std = np.std(zcr)

                # Energy
                energy = librosa.feature.rms(y=y_audio)
                energy_mean = np.mean(energy)
                energy_std = np.std(energy)

                features = np.concatenate([
                    stft_mean,
                    mfcc_mean,
                    chroma_mean,
                    spectral_contrast_mean,
                    np.array([zcr_mean, zcr_std]),
                    np.array([energy_mean, energy_std])
                ])

                X_rows.append(features)
                y_rows.append(genre_to_id[genre])
            except Exception as e:
                print(f" {file}: {e}")
                continue

    X = np.array(X_rows)
    y = np.array(y_rows)
    print(f"\nExtracted {len(X)} samples, features per sample: {X.shape[1] if len(X)>0 else 0}")
    return X, y, genre_to_id


def load_training_data_mfcc_only(folderpath, sr: int = 8000, duration: float | None = 15.0):
    """
    Extract MFCC-only features (per-file 13-d mean). Return X, y, genre_to_id.
    For speed, downsample to 8kHz and take the first 15 seconds by default.
    """
    print("Extracting MFCC features (MFCC-only, fast mode)...")
    genre_to_id = {genre: idx for idx, genre in enumerate(GENRES)}

    X_rows = []
    y_rows = []

    for genre in GENRES:
        genre_folder = os.path.join(folderpath, genre)
        if not os.path.exists(genre_folder):
            continue
        for file in os.listdir(genre_folder):
            if not file.endswith('.au'):
                continue
            file_path = os.path.join(genre_folder, file)
            try:
                y_audio, sr_loaded = librosa.core.load(file_path, sr=sr, mono=True, duration=duration)
                if y_audio is None or len(y_audio) == 0:
                    print(f"{file}: empty audio, skipped")
                    continue

                mfcc = librosa.feature.mfcc(y=y_audio, sr=sr if sr is not None else sr_loaded, n_mfcc=13)
                mfcc_mean = np.mean(mfcc, axis=1)  # (13,)

                X_rows.append(mfcc_mean)
                y_rows.append(genre_to_id[genre])
            except Exception as e:
                print(f" {file}: {e}")
                continue

    X = np.array(X_rows)
    y = np.array(y_rows)
    print(f"\nMFCC extraction done: {len(X)} samples, feature dim {X.shape[1] if len(X)>0 else 0}")
    return X, y, genre_to_id


def extract_features_from_file(file_path):
    """
    Extract all features from a single test file (for inference)
    
    Args:
        file_path: path to the audio file
    
    Returns:
        features: feature vector (1D numpy array)
    """
    try:
        y, sr = librosa.load(file_path, sr=None)
        
        # STFT
        stft = librosa.stft(y=y)
        stft_amp = np.abs(stft)
        stft_mean = np.mean(stft_amp, axis=1)
        
        # MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        
        # Chroma
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        
        # Spectral Contrast
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        spectral_contrast_mean = np.mean(spectral_contrast, axis=1)
        
        # Zero-Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y=y)
        zcr_mean = np.mean(zcr)
        zcr_std = np.std(zcr)
        
        # Energy
        energy = librosa.feature.rms(y=y)
        energy_mean = np.mean(energy)
        energy_std = np.std(energy)
        
        # Concatenate all features
        features = np.concatenate([
            stft_mean,
            mfcc_mean,
            chroma_mean,
            spectral_contrast_mean,
            np.array([zcr_mean, zcr_std]),
            np.array([energy_mean, energy_std])
        ])
        
        return features
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None


def load_test_data(test_folder_path, test_list_file):
    """
    Load test data
    
    Args:
        test_folder_path: test data folder path
        test_list_file: path to file list
    
    Returns:
        X_test: feature matrix
        test_ids: test file ids
    """
    # Read test file list
    with open(test_list_file, 'r') as f:
        test_files = [line.strip() for line in f.readlines()]
    
    X_test = []
    test_ids = []
    
    print(f"\nExtracting test features ({len(test_files)} files)...")
    
    for file_name in test_files:
        file_path = os.path.join(test_folder_path, file_name)
        
        if not os.path.exists(file_path):
            print(f"Warning: file {file_path} does not exist")
            continue
        
        features = extract_features_from_file(file_path)
        if features is not None:
            X_test.append(features)
            test_ids.append(file_name)
    
    X_test = np.array(X_test)
    print(f"Successfully extracted features for {len(X_test)} test samples")
    
    return X_test, test_ids


def pca(X, n_components: int):
    """
    Perform PCA on X and return the projected data, and return the best n_components.
    Args:
    X: numpy array of shape (n_samples, n_features)
    n_components: int, number of principal components to keep
    Returns:
    X_pca: numpy array of shape (n_samples, n_components)
    """
    # Compute orthonormal vector matrix
    xByXTranspose = X @ X.T
    evals, evecs = np.linalg.eigh(xByXTranspose)
    num_evals = evals.shape[0]
    eigenvals = np.zeros( num_evals )
    eigenvecs = np.zeros( ( num_evals, num_evals ) )
    # Sort eigenvalues and eigenvectors by importance
    for i in range( num_evals ):
        eigenvals[i] = evals[num_evals - i - 1]
        eigenvecs[:,i] = evecs[:,num_evals - i - 1]
    # Multiply for PCA
    PCA_mat = X @ eigenvecs
    Output_mat = PCA_mat[:, 0:n_components]
    return Output_mat

def load_training_data_stft_with_labels(folderpath):
    """
    Load the data from different directories, collect per-file Short-Term Fourier Transform average amplitude over time slices
    Returns (data, labels) with hardcoded labels 1-10 (from Project2_3)
    Args:
        folderpath: path to data directories
    Returns:
        (numpy array of data, numpy array of labels)
    """
    data = []
    labels = []
    for file in os.listdir(folderpath + "/rock"):
        y, sr = librosa.load(folderpath + "/rock/" + file)
        stft = librosa.stft(y=y)
        # Get amplitudes
        data_amp = np.abs(stft)
        # Average amplitude over time frames
        data.append(np.mean(data_amp, 1))
        labels.append(1)
    for file in os.listdir(folderpath + "/reggae"):
        y, sr = librosa.load(folderpath + "/reggae/" + file)
        stft = librosa.stft(y=y)
        # Get amplitudes
        data_amp = np.abs(stft)
        # Average amplitude over time frames
        data.append(np.mean(data_amp, 1))
        labels.append(2)
    for file in os.listdir(folderpath + "/pop"):
        y, sr = librosa.load(folderpath + "/pop/" + file)
        stft = librosa.stft(y=y)
        # Get amplitudes
        data_amp = np.abs(stft)
        # Average amplitude over time frames
        data.append(np.mean(data_amp, 1))
        labels.append(3)
    for file in os.listdir(folderpath + "/metal"):
        y, sr = librosa.load(folderpath + "/metal/" + file)
        stft = librosa.stft(y=y)
        # Get amplitudes
        data_amp = np.abs(stft)
        # Average amplitude over time frames
        data.append(np.mean(data_amp, 1))
        labels.append(4)
    for file in os.listdir(folderpath + "/jazz"):
        y, sr = librosa.load(folderpath + "/jazz/" + file)
        stft = librosa.stft(y=y)
        # Get amplitudes
        data_amp = np.abs(stft)
        # Average amplitude over time frames
        data.append(np.mean(data_amp, 1))
        labels.append(5)
    for file in os.listdir(folderpath + "/hiphop"):
        y, sr = librosa.load(folderpath + "/hiphop/" + file)
        stft = librosa.stft(y=y)
        # Get amplitudes
        data_amp = np.abs(stft)
        # Average amplitude over time frames
        data.append(np.mean(data_amp, 1))
        labels.append(6)
    for file in os.listdir(folderpath + "/disco"):
        y, sr = librosa.load(folderpath + "/disco/" + file)
        stft = librosa.stft(y=y)
        # Get amplitudes
        data_amp = np.abs(stft)
        # Average amplitude over time frames
        data.append(np.mean(data_amp, 1))
        labels.append(7)
    for file in os.listdir(folderpath + "/country"):
        y, sr = librosa.load(folderpath + "/country/" + file)
        stft = librosa.stft(y=y)
        # Get amplitudes
        data_amp = np.abs(stft)
        # Average amplitude over time frames
        data.append(np.mean(data_amp, 1))
        labels.append(8)
    for file in os.listdir(folderpath + "/classical"):
        y, sr = librosa.load(folderpath + "/classical/" + file)
        stft = librosa.stft(y=y)
        # Get amplitudes
        data_amp = np.abs(stft)
        # Average amplitude over time frames
        data.append(np.mean(data_amp, 1))
        labels.append(9)
    for file in os.listdir(folderpath + "/blues"):
        y, sr = librosa.load(folderpath + "/blues/" + file)
        stft = librosa.stft(y=y)
        # Get amplitudes
        data_amp = np.abs(stft)
        # Average amplitude over time frames
        data.append(np.mean(data_amp, 1))
        labels.append(10)    
    return np.array(data), np.array(labels)


def load_training_data_mfcc_with_labels(folderpath):
    """
    Load the data from different directories, collect per-file Mel Frequency Cepstral Coefficients average over coefficients
    Returns (data, labels) with hardcoded labels 1-10 (from Project2_3)
    Args:
        folderpath: path to data directories
    Returns:
        (numpy array of data, numpy array of labels)
    """
    # 13 is good number of coefficients
    data = []
    labels = []
    for file in os.listdir(folderpath + "/rock"):
        y, sr = librosa.load(folderpath + "/rock/" + file)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        # Average mfccs
        data.append(np.mean(mfcc, 1))
        labels.append(1)
    for file in os.listdir(folderpath + "/reggae"):
        y, sr = librosa.load(folderpath + "/reggae/" + file)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        # Average mfccs
        data.append(np.mean(mfcc, 1))
        labels.append(2)
    for file in os.listdir(folderpath + "/pop"):
        y, sr = librosa.load(folderpath + "/pop/" + file)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        # Average mfccs
        data.append(np.mean(mfcc, 1))
        labels.append(3)
    for file in os.listdir(folderpath + "/metal"):
        y, sr = librosa.load(folderpath + "/metal/" + file)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        # Average mfccs
        data.append(np.mean(mfcc, 1))
        labels.append(4)
    for file in os.listdir(folderpath + "/jazz"):
        y, sr = librosa.load(folderpath + "/jazz/" + file)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        # Average mfccs
        data.append(np.mean(mfcc, 1))
        labels.append(5)
    for file in os.listdir(folderpath + "/hiphop"):
        y, sr = librosa.load(folderpath + "/hiphop/" + file)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        # Average mfccs
        data.append(np.mean(mfcc, 1))
        labels.append(6)
    for file in os.listdir(folderpath + "/disco"):
        y, sr = librosa.load(folderpath + "/disco/" + file)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        # Average mfccs
        data.append(np.mean(mfcc, 1))
        labels.append(7)
    for file in os.listdir(folderpath + "/country"):
        y, sr = librosa.load(folderpath + "/country/" + file)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        # Average mfccs
        data.append(np.mean(mfcc, 1))
        labels.append(8)
    for file in os.listdir(folderpath + "/classical"):
        y, sr = librosa.load(folderpath + "/classical/" + file)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        # Average mfccs
        data.append(np.mean(mfcc, 1))
        labels.append(9)
    for file in os.listdir(folderpath + "/blues"):
        y, sr = librosa.load(folderpath + "/blues/" + file)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        # Average mfccs
        data.append(np.mean(mfcc, 1))
        labels.append(10)  
    return np.array(data), np.array(labels)


def load_training_data_cf_with_labels(folderpath):
    """
    Load the data from different directories, collect per-file Chroma feature average over chroma bin
    Returns (data, labels) with hardcoded labels 1-10 (from Project2_3)
    Args:
        folderpath: path to data directories
    Returns:
        (numpy array of data, numpy array of labels)
    """
    data = []
    labels = []
    for file in os.listdir(folderpath + "/rock"):
        y, sr = librosa.load(folderpath + "/rock/" + file)
        cf = librosa.feature.chroma_stft(y=y, sr=sr)
        # Average each chroma bin
        data.append(np.mean(cf, 1))
        labels.append(1)
    for file in os.listdir(folderpath + "/reggae"):
        y, sr = librosa.load(folderpath + "/reggae/" + file)
        cf = librosa.feature.chroma_stft(y=y, sr=sr)
        # Average each chroma bin
        data.append(np.mean(cf, 1))
        labels.append(2)
    for file in os.listdir(folderpath + "/pop"):
        y, sr = librosa.load(folderpath + "/pop/" + file)
        cf = librosa.feature.chroma_stft(y=y, sr=sr)
        # Average each chroma bin
        data.append(np.mean(cf, 1))
        labels.append(3)
    for file in os.listdir(folderpath + "/metal"):
        y, sr = librosa.load(folderpath + "/metal/" + file)
        cf = librosa.feature.chroma_stft(y=y, sr=sr)
        # Average each chroma bin
        data.append(np.mean(cf, 1))
        labels.append(4)
    for file in os.listdir(folderpath + "/jazz"):
        y, sr = librosa.load(folderpath + "/jazz/" + file)
        cf = librosa.feature.chroma_stft(y=y, sr=sr)
        # Average each chroma bin
        data.append(np.mean(cf, 1))
        labels.append(5)
    for file in os.listdir(folderpath + "/hiphop"):
        y, sr = librosa.load(folderpath + "/hiphop/" + file)
        cf = librosa.feature.chroma_stft(y=y, sr=sr)
        # Average each chroma bin
        data.append(np.mean(cf, 1))
        labels.append(6)
    for file in os.listdir(folderpath + "/disco"):
        y, sr = librosa.load(folderpath + "/disco/" + file)
        cf = librosa.feature.chroma_stft(y=y, sr=sr)
        # Average each chroma bin
        data.append(np.mean(cf, 1))
        labels.append(7)
    for file in os.listdir(folderpath + "/country"):
        y, sr = librosa.load(folderpath + "/country/" + file)
        cf = librosa.feature.chroma_stft(y=y, sr=sr)
        # Average each chroma bin
        data.append(np.mean(cf, 1))
        labels.append(8)
    for file in os.listdir(folderpath + "/classical"):
        y, sr = librosa.load(folderpath + "/classical/" + file)
        cf = librosa.feature.chroma_stft(y=y, sr=sr)
        # Average each chroma bin
        data.append(np.mean(cf, 1))
        labels.append(9)
    for file in os.listdir(folderpath + "/blues"):
        y, sr = librosa.load(folderpath + "/blues/" + file)
        cf = librosa.feature.chroma_stft(y=y, sr=sr)
        # Average each chroma bin
        data.append(np.mean(cf, 1))  
        labels.append(10)
    return np.array(data), np.array(labels)


def load_training_data_sc_with_labels(folderpath):
    """
    Load the data from different directories, collect per-file spectral contrast average over time
    Returns (data, labels) with hardcoded labels 1-10 (from Project2_3)
    Args:
        folderpath: path to data directories
    Returns:
        (numpy array of data, numpy array of labels)
    """
    data = []
    labels = []
    for file in os.listdir(folderpath + "/rock"):
        y, sr = librosa.load(folderpath + "/rock/" + file)
        sc = librosa.feature.spectral_contrast(y=y, sr=sr)
        # Average over time
        data.append(np.mean(sc, 1))
        labels.append(1)
    for file in os.listdir(folderpath + "/reggae"):
        y, sr = librosa.load(folderpath + "/reggae/" + file)
        sc = librosa.feature.spectral_contrast(y=y, sr=sr)
        # Average over time
        data.append(np.mean(sc, 1))
        labels.append(2)
    for file in os.listdir(folderpath + "/pop"):
        y, sr = librosa.load(folderpath + "/pop/" + file)
        sc = librosa.feature.spectral_contrast(y=y, sr=sr)
        # Average over time
        data.append(np.mean(sc, 1))
        labels.append(3)
    for file in os.listdir(folderpath + "/metal"):
        y, sr = librosa.load(folderpath + "/metal/" + file)
        sc = librosa.feature.spectral_contrast(y=y, sr=sr)
        # Average over time
        data.append(np.mean(sc, 1))
        labels.append(4)
    for file in os.listdir(folderpath + "/jazz"):
        y, sr = librosa.load(folderpath + "/jazz/" + file)
        sc = librosa.feature.spectral_contrast(y=y, sr=sr)
        # Average over time
        data.append(np.mean(sc, 1))
        labels.append(5)
    for file in os.listdir(folderpath + "/hiphop"):
        y, sr = librosa.load(folderpath + "/hiphop/" + file)
        sc = librosa.feature.spectral_contrast(y=y, sr=sr)
        # Average over time
        data.append(np.mean(sc, 1))
        labels.append(6)
    for file in os.listdir(folderpath + "/disco"):
        y, sr = librosa.load(folderpath + "/disco/" + file)
        sc = librosa.feature.spectral_contrast(y=y, sr=sr)
        # Average over time
        data.append(np.mean(sc, 1))
        labels.append(7)
    for file in os.listdir(folderpath + "/country"):
        y, sr = librosa.load(folderpath + "/country/" + file)
        sc = librosa.feature.spectral_contrast(y=y, sr=sr)
        # Average over time
        data.append(np.mean(sc, 1))
        labels.append(8)
    for file in os.listdir(folderpath + "/classical"):
        y, sr = librosa.load(folderpath + "/classical/" + file)
        sc = librosa.feature.spectral_contrast(y=y, sr=sr)
        # Average over time
        data.append(np.mean(sc, 1))
        labels.append(9)
    for file in os.listdir(folderpath + "/blues"):
        y, sr = librosa.load(folderpath + "/blues/" + file)
        sc = librosa.feature.spectral_contrast(y=y, sr=sr)
        # Average over time
        data.append(np.mean(sc, 1))
        labels.append(10)
    return np.array(data), np.array(labels)


def load_training_data_zcr_with_labels(folderpath):
    """
    Load the data from different directories, collect per-file zero-crossing rate over time
    Returns (data, labels) with hardcoded labels 1-10 (from Project2_3)
    Note: This version returns raw ZCR (2D), different from load_training_data_zcr which returns [mean, std]
    Args:
        folderpath: path to data directories
    Returns:
        (numpy array of data, numpy array of labels)
    """
    data = []
    labels = []
    for file in os.listdir(folderpath + "/rock"):
        y, sr = librosa.load(folderpath + "/rock/" + file)
        zcr = librosa.feature.zero_crossing_rate(y=y)
        # Add RMS to data
        data.append(zcr)
        labels.append(1)
    for file in os.listdir(folderpath + "/reggae"):
        y, sr = librosa.load(folderpath + "/reggae/" + file)
        zcr = librosa.feature.zero_crossing_rate(y=y)
        # Add RMS to data
        data.append(zcr)
        labels.append(2)
    for file in os.listdir(folderpath + "/pop"):
        y, sr = librosa.load(folderpath + "/pop/" + file)
        zcr = librosa.feature.zero_crossing_rate(y=y)
        # Add RMS to data
        data.append(zcr)
        labels.append(3)
    for file in os.listdir(folderpath + "/metal"):
        y, sr = librosa.load(folderpath + "/metal/" + file)
        zcr = librosa.feature.zero_crossing_rate(y=y)
        # Add RMS to data
        data.append(zcr)
        labels.append(4)
    for file in os.listdir(folderpath + "/jazz"):
        y, sr = librosa.load(folderpath + "/jazz/" + file)
        zcr = librosa.feature.zero_crossing_rate(y=y)
        # Add RMS to data
        data.append(zcr)
        labels.append(5)
    for file in os.listdir(folderpath + "/hiphop"):
        y, sr = librosa.load(folderpath + "/hiphop/" + file)
        zcr = librosa.feature.zero_crossing_rate(y=y)
        # Add RMS to data
        data.append(zcr)
        labels.append(6)
    for file in os.listdir(folderpath + "/disco"):
        y, sr = librosa.load(folderpath + "/disco/" + file)
        zcr = librosa.feature.zero_crossing_rate(y=y)
        # Add RMS to data
        data.append(zcr)
        labels.append(7)
    for file in os.listdir(folderpath + "/country"):
        y, sr = librosa.load(folderpath + "/country/" + file)
        zcr = librosa.feature.zero_crossing_rate(y=y)
        # Add RMS to data
        data.append(zcr)
        labels.append(8)
    for file in os.listdir(folderpath + "/classical"):
        y, sr = librosa.load(folderpath + "/classical/" + file)
        zcr = librosa.feature.zero_crossing_rate(y=y)
        # Add RMS to data
        data.append(zcr)
        labels.append(9)
    for file in os.listdir(folderpath + "/blues"):
        y, sr = librosa.load(folderpath + "/blues/" + file)
        zcr = librosa.feature.zero_crossing_rate(y=y)
        # Add RMS to data
        data.append(zcr)  
        labels.append(10)
    return np.array(data), np.array(labels)


def load_training_data_energy_with_labels(folderpath):
    """
    Load the data from different directories, collect per-file energy over time
    Returns (data, labels) with hardcoded labels 1-10 (from Project2_3)
    Note: This version returns energy.flatten(), different from load_training_data_energy which returns [mean, std]
    Args:
        folderpath: path to data directories
    Returns:
        (numpy array of data, numpy array of labels)
    """
    data = []
    labels = []
    for file in os.listdir(folderpath + "/rock"):
        y, sr = librosa.load(folderpath + "/rock/" + file)
        energy = librosa.feature.rms(y=y)
        # Add RMS to data
        data.append(energy.flatten())
        labels.append(1)
    for file in os.listdir(folderpath + "/reggae"):
        y, sr = librosa.load(folderpath + "/reggae/" + file)
        energy = librosa.feature.rms(y=y)
        # Add RMS to data
        data.append(energy.flatten())
        labels.append(2)
    for file in os.listdir(folderpath + "/pop"):
        y, sr = librosa.load(folderpath + "/pop/" + file)
        energy = librosa.feature.rms(y=y)
        # Add RMS to data
        data.append(energy.flatten())
        labels.append(3)
    for file in os.listdir(folderpath + "/metal"):
        y, sr = librosa.load(folderpath + "/metal/" + file)
        energy = librosa.feature.rms(y=y)
        # Add RMS to data
        data.append(energy.flatten())
        labels.append(4)
    for file in os.listdir(folderpath + "/jazz"):
        y, sr = librosa.load(folderpath + "/jazz/" + file)
        energy = librosa.feature.rms(y=y)
        # Add RMS to data
        data.append(energy.flatten())
        labels.append(5)
    for file in os.listdir(folderpath + "/hiphop"):
        y, sr = librosa.load(folderpath + "/hiphop/" + file)
        energy = librosa.feature.rms(y=y)
        # Add RMS to data
        data.append(energy.flatten())
        labels.append(6)
    for file in os.listdir(folderpath + "/disco"):
        y, sr = librosa.load(folderpath + "/disco/" + file)
        energy = librosa.feature.rms(y=y)
        # Add RMS to data
        data.append(energy.flatten())
        labels.append(7)
    for file in os.listdir(folderpath + "/country"):
        y, sr = librosa.load(folderpath + "/country/" + file)
        energy = librosa.feature.rms(y=y)
        # Add RMS to data
        data.append(energy.flatten())
        labels.append(8)
    for file in os.listdir(folderpath + "/classical"):
        y, sr = librosa.load(folderpath + "/classical/" + file)
        energy = librosa.feature.rms(y=y)
        # Add RMS to data
        data.append(energy.flatten())
        labels.append(9)
    for file in os.listdir(folderpath + "/blues"):
        y, sr = librosa.load(folderpath + "/blues/" + file)
        energy = librosa.feature.rms(y=y)
        # Add RMS to data
        data.append(energy.flatten())  
        labels.append(10)
    return np.array(data), np.array(labels)


def load_training_data_all(folderpath):
    """
    Load the data from different directories, collect per-file data
    Combines STFT, MFCC, Chroma, and Spectral Contrast features (from Project2_3)
    Args:
        folderpath: path to data directories
    Returns:
        (numpy array of data, numpy array of labels)
    """
    data_stft, labels = load_training_data_stft_with_labels(folderpath)
    data_mfcc, labels_unused = load_training_data_mfcc_with_labels(folderpath)
    data_cf, labels_unused = load_training_data_cf_with_labels(folderpath)
    data_sc, labels_unused = load_training_data_sc_with_labels(folderpath)
    return np.hstack((data_stft, data_mfcc, data_cf, data_sc)), labels
