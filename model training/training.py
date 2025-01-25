import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    precision_recall_curve,
    auc,
    roc_auc_score,
    precision_recall_fscore_support,
)
import matplotlib.pyplot as plt


def load_data(filepath: str) -> pd.DataFrame:
    """Load dataset from a CSV file."""
    return pd.read_csv(filepath, index_col=0)


def split_and_scale_data(data: pd.DataFrame):
    """Split data into train/validation sets and scale features."""
    X = data.drop(columns=["label"])
    y = data["label"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    return X_train_scaled, X_val_scaled, y_train, y_val, scaler


def train_and_evaluate(X_train, X_val, y_train, y_val):
    """Train and evaluate a Logistic Regression model using a train/validation split."""
    model = LogisticRegression(class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]

    print("Classification Report:")
    print(classification_report(y_val, y_pred))

    precision, recall, _ = precision_recall_curve(y_val, y_pred_proba)
    auc_pr = auc(recall, precision)
    print(f"AUC-PR: {auc_pr:.4f}")

    plot_precision_recall_curve(precision, recall, auc_pr)

    return model


def cross_validation_evaluation(data: pd.DataFrame):
    """Perform StratifiedKFold cross-validation and evaluate metrics."""
    X = data.drop(columns=["label"])
    y = data["label"]

    skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

    auc_scores = []
    auc_pr_scores = []
    precision_scores = []
    recall_scores = []
    fscore_scores = []

    model = LogisticRegression(class_weight="balanced", random_state=42)

    for train_index, val_index in skf.split(X, y):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        model.fit(X_train_scaled, y_train)

        y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
        auc_scores.append(roc_auc_score(y_val, y_pred_proba))

        precision, recall, _ = precision_recall_curve(y_val, y_pred_proba)
        auc_pr_scores.append(auc(recall, precision))

        precision_class, recall_class, fscore, _ = precision_recall_fscore_support(
            y_val, y_pred_proba > 0.5
        )
        precision_scores.append(precision_class[1])
        recall_scores.append(recall_class[1])
        fscore_scores.append(fscore[1])

    print("Cross-validation metrics:")
    print(f"AUC-ROC scores: {auc_scores}")
    print(f"Mean AUC-ROC: {np.mean(auc_scores):.4f}")
    print(f"AUC-PR scores: {auc_pr_scores}")
    print(f"Mean AUC-PR: {np.mean(auc_pr_scores):.4f}")
    print(f"Precision: {np.mean(precision_scores):.4f}")
    print(f"Recall: {np.mean(recall_scores):.4f}")
    print(f"F1-Score: {np.mean(fscore_scores):.4f}")

    return model


def train_on_full_data(data: pd.DataFrame):
    """Train a Logistic Regression model on the full dataset."""
    X = data.drop(columns=["label"])
    y = data["label"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(class_weight="balanced", random_state=42)
    model.fit(X_scaled, y)

    joblib.dump(model, "logistic_regression_model.pkl")
    print("Model saved to logistic_regression_model.pkl")

    y_pred = model.predict(X_scaled)
    y_pred_proba = model.predict_proba(X_scaled)[:, 1]

    print("Classification Report:")
    print(classification_report(y, y_pred))

    precision, recall, _ = precision_recall_curve(y, y_pred_proba)
    auc_pr = auc(recall, precision)
    print(f"AUC-PR: {auc_pr:.4f}")

    plot_precision_recall_curve(precision, recall, auc_pr)

    return model


def plot_precision_recall_curve(precision, recall, auc_pr):
    """Plot the Precision-Recall curve."""
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f"AUC-PR = {auc_pr:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    filepath = "./balanced_data.csv"
    data = load_data(filepath)

    X_train, X_val, y_train, y_val, _ = split_and_scale_data(data)
    train_and_evaluate(X_train, X_val, y_train, y_val)

    cross_validation_evaluation(data)

    train_on_full_data(data)
