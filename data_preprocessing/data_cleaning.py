import numpy as np

def drop_missing_values(data, dataset_name="dataset"):
    """
    Drops rows with missing values and reports the percentage of rows removed.

    :param data: DataFrame to clean
    :param dataset_name: Name of the dataset (for reporting purposes)
    :return: Tuple (cleaned DataFrame, percentage of rows removed)
    """
    initial_rows = len(data)
    cleaned_data = data.replace([np.inf, -np.inf], np.nan).dropna()
    removed_rows = initial_rows - len(cleaned_data)
    percentage_removed = (removed_rows / initial_rows) * 100

    print(f"[{dataset_name}] Dropped rows with missing values: {removed_rows} ({percentage_removed:.2f}%)")
    return cleaned_data, percentage_removed

def drop_zero_columns(data, dataset_name="dataset"):
    """
    Drops columns where all values are zero and reports the names of the removed columns.

    :param data: DataFrame to clean
    :param dataset_name: Name of the dataset (for reporting purposes)
    :return: Tuple (cleaned DataFrame, list of dropped column names)
    """
    zero_columns = data.columns[(data == 0).all()]
    cleaned_data = data.drop(columns=zero_columns)

    print(f"[{dataset_name}] Dropped zero-value columns: {list(zero_columns)}")
    return cleaned_data, list(zero_columns)

def clean_datasets(positive_data, unlabelled_data):
    """
    Cleans both positive and unlabelled datasets by:
    - Dropping rows with missing values.
    - Dropping columns where all values are zero.
    - Providing summary reports for each dataset.

    :param positive_data: DataFrame containing positive class samples
    :param unlabelled_data: DataFrame containing unlabelled class samples
    :return: Dictionary containing cleaned datasets and a summary report
    """
    print("Cleaning Positive Dataset:")
    positive_data_clean, pos_missing_percentage = drop_missing_values(positive_data, dataset_name="Positive Dataset")
    positive_data_clean, pos_zero_columns = drop_zero_columns(positive_data_clean, dataset_name="Positive Dataset")

    print("\nCleaning Unlabelled Dataset:")
    unlabelled_data_clean, unlabelled_missing_percentage = drop_missing_values(unlabelled_data, dataset_name="Unlabelled Dataset")
    unlabelled_data_clean, unlabelled_zero_columns = drop_zero_columns(unlabelled_data_clean, dataset_name="Unlabelled Dataset")

    print(f"Max value in X_train: {positive_data_clean.max().max()}")
    print(f"Min value in X_train: {positive_data_clean.min().min()}")

    return {
        "positive_data_clean": positive_data_clean,
        "unlabelled_data_clean": unlabelled_data_clean,
        "summary": {
            "positive": {
                "missing_percentage": pos_missing_percentage,
                "dropped_zero_columns": pos_zero_columns
            },
            "unlabelled": {
                "missing_percentage": unlabelled_missing_percentage,
                "dropped_zero_columns": unlabelled_zero_columns
            }
        }
    }

def save_cleaned_data(data, filename):
    """
    Saves the cleaned dataset to a CSV file.

    :param data: DataFrame to save
    :param filename: String name of the file to save the data to
    """
    data.to_csv(filename, index=True)
    print(f"Cleaned data saved to {filename}")

