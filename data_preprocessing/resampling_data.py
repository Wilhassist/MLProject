import pandas as pd
from sklearn.utils import resample

from data_cleaning import clean_datasets

# Function to downsample the majority class
def downsample_majority_class(positive_data, unlabelled_data):
    """
    Downsample the majority class (unlabelled_data) to match the size of the minority class (positive_data).
    
    :param positive_data: DataFrame containing the positive class (minority)
    :param unlabelled_data: DataFrame containing the unlabelled class (majority)
    :return: DataFrame containing the downsampled dataset
    """
    minority_size = len(positive_data)
    
    # Downsample the majority class (unlabelled data)
    unlabelled_downsampled = unlabelled_data.sample(n=minority_size, random_state=42)
    
    return unlabelled_downsampled

# Function to oversample the minority class
def oversample_minority_class(positive_data, unlabelled_data):
    """
    Oversample the minority class (positive_data) to match the size of the majority class (unlabelled_data).
    
    :param positive_data: DataFrame containing the positive class (minority)
    :param unlabelled_data: DataFrame containing the unlabelled class (majority)
    :return: DataFrame containing the oversampled dataset
    """
    majority_size = len(unlabelled_data)
    
    # Upsample the minority class (positive data)
    positive_oversampled = resample(positive_data, 
                                    replace=True,    # sample with replacement
                                    n_samples=majority_size,  # match the majority class size
                                    random_state=42)
    
    return positive_oversampled

# Function for hybrid sampling (downsample majority and then oversample minority)
def hybrid_sampling(positive_data, unlabelled_data, ratio=1.0):
    """
    Apply hybrid sampling by first downsampling the majority class and then oversampling the minority class.
    
    :param positive_data: DataFrame containing the positive class (minority)
    :param unlabelled_data: DataFrame containing the unlabelled class (majority)
    :param ratio: The ratio of oversampling (e.g., 1.0 means oversample to the same size as the majority class)
    :return: DataFrame containing the combined balanced dataset
    """
    # Downsample the majority class (unlabelled data)
    minority_size = len(positive_data)
    unlabelled_downsampled = unlabelled_data.sample(n=int(minority_size * ratio), random_state=42)
    
    # Oversample the minority class (positive data)
    majority_size = len(unlabelled_downsampled)
    positive_oversampled = resample(positive_data, 
                                    replace=True, 
                                    n_samples=majority_size,  
                                    random_state=42)
    
    return positive_oversampled, unlabelled_downsampled

# Function to join positive and unlabelled data after performing sampling
def join_datasets(positive_data, unlabelled_data, sampling_type="downsample", ratio=1.0):
    """
    Join positive and unlabelled datasets after performing either downsampling, oversampling, or hybrid sampling.
    
    :param positive_data: DataFrame containing the positive class (minority)
    :param unlabelled_data: DataFrame containing the unlabelled class (majority)
    :param sampling_type: Type of sampling to use ('downsample', 'oversample', or 'hybrid')
    :param ratio: The ratio for hybrid sampling (default 1.0)
    :return: DataFrame containing the combined and balanced dataset
    """

    # Clean the datasets
    cleaning_results = clean_datasets(positive_data, unlabelled_data)

    # Access cleaned data
    positive_data_clean = cleaning_results["positive_data_clean"]
    unlabelled_data_clean = cleaning_results["unlabelled_data_clean"]

    if sampling_type == "downsample":
        unlabelled_data = downsample_majority_class(positive_data_clean, unlabelled_data_clean)
    elif sampling_type == "oversample":
        positive_data = oversample_minority_class(positive_data_clean, unlabelled_data_clean)
    elif sampling_type == "hybrid":
        positive_data, unlabelled_data = hybrid_sampling(positive_data_clean, unlabelled_data_clean, ratio)
    else:
        raise ValueError("Invalid sampling_type. Choose either 'downsample', 'oversample', or 'hybrid'.")

    positive_data = positive_data.copy()
    unlabelled_data = unlabelled_data.copy()
    
    # Add labels to both datasets
    positive_data['label'] = 1
    unlabelled_data['label'] = 0
    
    # Combine the positive and adjusted unlabelled data
    balanced_data = pd.concat([positive_data, unlabelled_data], axis=0)
    
    # Shuffle the combined data to randomize row order
    balanced_data = balanced_data.sample(frac=1, random_state=42)
    
    return balanced_data

# Load the datasets
positive_data = pd.read_csv("../data/training_pos_features.csv", index_col=0)
unlabelled_data = pd.read_csv("../data/training_others_features.csv", index_col=0)

def data_downsampled(positive_data = positive_data, unlabelled_data = unlabelled_data):
    balanced_data_downsampled = join_datasets(positive_data, unlabelled_data, sampling_type="downsample")
    print(f"Balanced dataset (downsampled): {len(balanced_data_downsampled)} samples")
    return balanced_data_downsampled

def data_oversampled(positive_data = positive_data, unlabelled_data = unlabelled_data):
    balanced_data_oversampled = join_datasets(positive_data, unlabelled_data, sampling_type="oversample")
    print(f"Balanced dataset (oversampled): {len(balanced_data_oversampled)} samples")
    return balanced_data_oversampled

def data_hybridsampled(positive_data = positive_data, unlabelled_data = unlabelled_data, ratio = 0.5):
    balanced_data_hybrid = join_datasets(positive_data, unlabelled_data, sampling_type="hybrid", ratio=ratio)
    print(f"Balanced dataset (hybrid): {len(balanced_data_hybrid)} samples")
    return balanced_data_hybrid

data = data_downsampled()
data.to_csv("balanced_data.csv", index=True)
