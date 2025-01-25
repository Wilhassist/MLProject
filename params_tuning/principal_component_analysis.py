from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd

def apply_pca(X):

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=None)  # n_components=None retains all components
    pca.fit(X_scaled)

    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = explained_variance_ratio.cumsum()

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
    plt.axhline(y=0.95, color='r', linestyle='--')  # Threshold for 95% variance
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA Explained Variance')
    plt.grid()
    plt.show()

    n_components = next(i for i, total in enumerate(cumulative_variance) if total >= 0.95) + 1
    print(f"Number of components to retain 95% variance: {n_components}")

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)


    X_pca_df = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(n_components)])
    return X_pca_df