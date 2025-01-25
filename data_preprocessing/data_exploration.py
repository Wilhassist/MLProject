import matplotlib.pyplot as plt
import seaborn as sns

def gx_feature_correlation(data, feature_regex):

    # Filter columns starting with "G10"
    features = data.filter(regex=feature_regex)

    # Compute the correlation matrix
    correlation_matrix = features.corr().abs()

    # Plot the heatmap
    plt.figure(figsize=(10, 8))  # Adjust the size as needed
    sns.heatmap(correlation_matrix, annot=False, cmap="coolwarm", cbar=True)
    plt.title("Correlation Matrix of G Features")
    plt.show()

# Function to plot side-by-side box plots
def plot_features_side_by_side(data, start_index, end_index, plot_type='boxplot'):
    """
    Plots box plots or histograms of numeric features side by side.
    
    Parameters:
    - data: DataFrame containing the data.
    - start_index: Start index of features to plot.
    - end_index: End index of features to plot.
    - plot_type: 'boxplot' or 'histplot' for the type of plot.
    """
    numeric_cols = [col for col in data.columns if data[col].dtype in ['float64', 'int64']]
    selected_cols = numeric_cols[start_index:end_index]
    
    num_plots = len(selected_cols)
    num_cols = min(num_plots, 3)  # Number of columns per row
    num_rows = (num_plots // num_cols) + (num_plots % num_cols > 0)
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 5))
    axes = axes.flatten() if num_plots > 1 else [axes]  # Ensure axes is iterable
    
    for i, col in enumerate(selected_cols):
        if plot_type == 'boxplot':
            sns.boxplot(x=data[col], ax=axes[i])
        elif plot_type == 'histplot':
            sns.histplot(data[col], kde=True, ax=axes[i])
        axes[i].set_title(f"{plot_type.capitalize()} of {col}")
    
    # Turn off unused subplots
    for j in range(len(selected_cols), len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.show()



def plot_features_by_type(data, feature_type, batch_size=10, plot_type='boxplot'):
    """
    Plots numeric features by feature type in batches of a given size.
    
    Parameters:
    - data: DataFrame containing the data.
    - feature_type: String indicating the prefix of the feature columns (e.g., "G1").
    - batch_size: Number of features to plot in each batch (default: 10).
    - plot_type: 'boxplot' or 'histplot' for the type of plot (default: 'boxplot').
    """
    # Filter columns based on feature type
    selected_cols = [col for col in data.columns if col.startswith(feature_type)]
    
    if not selected_cols:
        print(f"No columns found for feature type: {feature_type}")
        return

    num_features = len(selected_cols)
    print(f"Found {num_features} features starting with '{feature_type}'.")
    
    # Plot in batches
    for start_index in range(0, num_features, batch_size):
        end_index = min(start_index + batch_size, num_features)
        print(f"Plotting features {start_index + 1} to {end_index}...")
        plot_features_side_by_side(data, start_index=start_index, end_index=end_index, plot_type=plot_type)

# Example usage:
# Plot G1 features in batches of 10 using box plots
# plot_features_by_type(positive_data, feature_type="G1", batch_size=10, plot_type='boxplot')

# Plot G2 features in batches of 10 using histograms
# plot_features_by_type(positive_data, feature_type="G1", batch_size=10, plot_type='histplot')

"""
# Question 2: How much data do I have and in what proportion?
pos_amount = positive_data.shape[0]
neg_amount = negative_data.shape[0]

sample_amount = pos_amount + neg_amount

print(f"Sample amount of data {sample_amount}")

pos_proportion = pos_amount * 100 / sample_amount 
neg_proportion = neg_amount * 100 / sample_amount

print(f"Positive proportion {pos_proportion}")
print(f"Negative proportion {neg_proportion}")

# Answer : Small Dataset with 
# - Sample amount of data 220022
# - Positive proportion 9.090909090909092 (20002)
# - Negative proportion 90.9090909090909 (200020)

"""