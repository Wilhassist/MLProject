import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, auc, classification_report,roc_auc_score, precision_recall_fscore_support

from resampling_data import data_downsampled

data = data_downsampled()

X = data.drop(columns=['label'])
y = data['label']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100, 
    max_depth=None, 
    class_weight='balanced', 
    random_state=42
)

# Stratified K-Fold Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
auc_scores = []
auc_pr_scores = []
precision_scores = []
recall_scores = []
fscore_scores = []

for train_index, test_index in skf.split(X_scaled, y):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    rf_model.fit(X_train, y_train)
    y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
    
    auc_score = roc_auc_score(y_test, y_pred_proba)
    auc_scores.append(auc_score)

    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    auc_pr = auc(recall, precision)
    auc_pr_scores.append(auc_pr)

    # Calculate Precision, Recall, and F1-Score
    precision, recall, fscore, _ = precision_recall_fscore_support(y_test, y_pred_proba > 0.5)
    precision_scores.append(precision[1])  # Precision for class 1 (effectors)
    recall_scores.append(recall[1])  # Recall for class 1 (effectors)
    fscore_scores.append(fscore[1])  # F1-score for class 1 (effectors)

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

# Train Final Model on Full Dataset
rf_model.fit(X_scaled, y)
y_pred = rf_model.predict(X_scaled)
y_pred_proba = rf_model.predict_proba(X_scaled)[:, 1]

# Classification Report
print("\nClassification Report:")
print(classification_report(y, y_pred))

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y, y_pred_proba)
final_auc_pr = auc(recall, precision)

import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f"AUC-PR = {final_auc_pr:.4f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve (Full Dataset)")
plt.legend()
plt.grid()
plt.show()
