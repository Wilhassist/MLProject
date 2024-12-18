import pandas as pd

positive_data = pd.read_csv("../data/training_pos_features.csv", index_col=0)
unlabelled_data = pd.read_csv("../data/training_others_features.csv", index_col=0)

# Determine the size of the minority class (positive class)
minority_size = len(positive_data)

# Downsample the majority class (unlabelled data)
unlabelled_downsampled = unlabelled_data.sample(n=minority_size, random_state=42)

positive_data['label'] = 1
unlabelled_downsampled['label'] = 0

# Combine the positive and downsampled unlabelled data
balanced_data = pd.concat([positive_data, unlabelled_downsampled], axis=0)

# Shuffle the combined data to randomize row order
balanced_data = balanced_data.sample(frac=1, random_state=42)

# Save the balanced dataset to a new CSV file
balanced_data.to_csv("balanced_data.csv", index=True)

# Output the number of samples in the new dataset
print(f"Total samples: {len(balanced_data)}")
print(f"Positive samples: {len(positive_data)}")
print(f"Downsampled unlabelled samples: {len(unlabelled_downsampled)}")
