import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_recall_curve, auc, roc_auc_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
positive_data = pd.read_csv("../data/training_pos_features.csv", index_col=0)
unlabelled_data = pd.read_csv("../data/training_others_features.csv", index_col=0)

# Label the data
positive_data['label'] = 1
unlabelled_data['label'] = 0

# Combine the positive and unlabelled data
data = pd.concat([positive_data, unlabelled_data], axis=0)

# Shuffle the data
data = data.sample(frac=1, random_state=42)

# Split the dataset into features and labels
X = data.drop(columns=["label"])  
y = data["label"]

# Initialize the model
model = LogisticRegression(class_weight="balanced", random_state=42)

# Set up 5-fold cross-validation
skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

# Lists to store results for each fold
auc_scores = []
auc_pr_scores = []
precision_scores = []
recall_scores = []
fscore_scores = []

# Cross-validation loop
for train_index, val_index in skf.split(X, y):
    X_train_cv, X_val_cv = X.iloc[train_index], X.iloc[val_index]
    y_train_cv, y_val_cv = y.iloc[train_index], y.iloc[val_index]

    # Scale the data
    scaler = StandardScaler()
    X_train_cv_scaled = scaler.fit_transform(X_train_cv)
    X_val_cv_scaled = scaler.transform(X_val_cv)

    # Train the model
    model.fit(X_train_cv_scaled, y_train_cv)

    # Make predictions and calculate probabilities
    y_pred_proba_cv = model.predict_proba(X_val_cv_scaled)[:, 1]

    # Calculate AUC-ROC
    auc_score = roc_auc_score(y_val_cv, y_pred_proba_cv)
    auc_scores.append(auc_score)

    # Calculate AUC-PR
    precision, recall, _ = precision_recall_curve(y_val_cv, y_pred_proba_cv)
    auc_pr_score = auc(recall, precision)
    auc_pr_scores.append(auc_pr_score)

    # Calculate Precision, Recall, and F1-Score
    precision, recall, fscore, _ = precision_recall_fscore_support(y_val_cv, y_pred_proba_cv > 0.5)
    precision_scores.append(precision[1])  # Precision for class 1 (effectors)
    recall_scores.append(recall[1])  # Recall for class 1 (effectors)
    fscore_scores.append(fscore[1])  # F1-score for class 1 (effectors)

# Print the results for each fold
print(f"AUC-ROC scores for each fold: {auc_scores}")
print(f"Mean AUC-ROC from cross-validation: {np.mean(auc_scores):.4f}")

print(f"AUC-PR scores for each fold: {auc_pr_scores}")
print(f"Mean AUC-PR from cross-validation: {np.mean(auc_pr_scores):.4f}")

print(f"Precision scores for each fold: {precision_scores}")
print(f"Mean Precision from cross-validation: {np.mean(precision_scores):.4f}")

print(f"Recall scores for each fold: {recall_scores}")
print(f"Mean Recall from cross-validation: {np.mean(recall_scores):.4f}")

print(f"F1-Score scores for each fold: {fscore_scores}")
print(f"Mean F1-Score from cross-validation: {np.mean(fscore_scores):.4f}")

# Train the model on the full dataset and save it
X_scaled = scaler.fit_transform(X)
model.fit(X_scaled, y)
joblib.dump(model, "logistic_regression_model.pkl")
print("Model saved to logistic_regression_model.pkl")

# Make predictions on the full dataset
y_pred = model.predict(X_scaled)
y_pred_proba = model.predict_proba(X_scaled)[:, 1]

# Evaluate the model on the full dataset
print("Classification Report:")
print(classification_report(y, y_pred))

# Calculate and plot AUC-PR for the full dataset
precision, recall, thresholds = precision_recall_curve(y, y_pred_proba)
auc_pr = auc(recall, precision)
print(f"AUC-PR on full dataset: {auc_pr:.4f}")

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f"AUC-PR = {auc_pr:.4f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.grid()
plt.show()

# Calculate and plot AUC-ROC for the full dataset
roc_auc = roc_auc_score(y, y_pred_proba)
print(f"AUC-ROC on full dataset: {roc_auc:.4f}")
