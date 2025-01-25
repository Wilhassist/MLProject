import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def select_k_best_features(X, y, k):
    """
    Select the top k features based on univariate statistical tests.
    
    :param X: DataFrame, feature matrix
    :param y: Series, target variable
    :param k: int, number of top features to select
    :return: DataFrame, selected features
    """
    selector = SelectKBest(score_func=f_classif, k=k)
    X_new = selector.fit_transform(X, y)
    mask = selector.get_support()  # Boolean mask of selected features
    selected_features = X.columns[mask]
    return X_new, selected_features

def recursive_feature_elimination(X, y, estimator=None, n_features=10):
    """
    Perform recursive feature elimination to select features.
    
    :param X: DataFrame, feature matrix
    :param y: Series, target variable
    :param estimator: model, base estimator (e.g., LogisticRegression)
    :param n_features: int, number of features to select
    :return: DataFrame, selected features
    """
    if estimator is None:
        estimator = LogisticRegression()
    rfe = RFE(estimator, n_features_to_select=n_features)
    X_new = rfe.fit_transform(X, y)
    mask = rfe.support_  # Boolean mask of selected features
    selected_features = X.columns[mask]
    return X_new, selected_features

def feature_importance_from_model(X, y, model=None):
    """
    Get feature importance scores from a model.
    
    :param X: DataFrame, feature matrix
    :param y: Series, target variable
    :param model: estimator, a model with `feature_importances_` or `coef_`
    :return: Series, feature importance scores
    """
    if model is None:
        model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance = model.coef_[0]
    else:
        raise ValueError("The model does not support feature importance.")
    feature_importance = pd.Series(importance, index=X.columns).sort_values(ascending=False)
    return feature_importance

def apply_pca(X, n_components=None):
    """
    Apply Principal Component Analysis (PCA) to reduce dimensionality.
    
    :param X: DataFrame, feature matrix
    :param n_components: int, number of components to keep (default=None keeps all components)
    :return: DataFrame, transformed features with reduced dimensionality, explained variance ratio
    """
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    explained_variance_ratio = pca.explained_variance_ratio_
    pca_columns = [f"PC{i+1}" for i in range(X_pca.shape[1])]
    X_pca_df = pd.DataFrame(X_pca, columns=pca_columns, index=X.index)
    return X_pca_df, explained_variance_ratio

def plot_correlation_matrix(data, threshold=0.8, figsize=(10, 8)):
    """
    Plot the correlation matrix of the dataset.
    
    :param data: DataFrame, feature matrix
    :param threshold: float, correlation threshold to highlight (default is 0.8)
    :param figsize: tuple, figure size for the plot
    :return: DataFrame, correlation matrix
    """
    corr_matrix = data.corr()

    # Identify highly correlated features
    high_corr_features = [
        (i, j, corr_matrix.loc[i, j])
        for i in corr_matrix.columns
        for j in corr_matrix.columns
        if i != j and abs(corr_matrix.loc[i, j]) > threshold
    ]

    if high_corr_features:
        print("Highly correlated features (threshold > {}):".format(threshold))
        for feature_pair in high_corr_features:
            print(f"{feature_pair[0]} <-> {feature_pair[1]}: {feature_pair[2]:.2f}")
    else:
        print("No highly correlated features found above the threshold.")

    return corr_matrix
