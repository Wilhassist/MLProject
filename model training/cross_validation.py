import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_recall_curve, auc, roc_auc_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import numpy as np

from data_preprocessing.resampling_data import data_oversampled

data = data_oversampled()

X = data.drop(columns=["label"])  
y = data["label"]

model = LogisticRegression(class_weight="balanced", random_state=42)

skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

auc_scores = []
auc_pr_scores = []
precision_scores = []
recall_scores = []
fscore_scores = []

for train_index, val_index in skf.split(X, y):
    X_train_cv, X_val_cv = X.iloc[train_index], X.iloc[val_index]
    y_train_cv, y_val_cv = y.iloc[train_index], y.iloc[val_index]

    scaler = StandardScaler()
    X_train_cv_scaled = scaler.fit_transform(X_train_cv)
    X_val_cv_scaled = scaler.transform(X_val_cv)

    model.fit(X_train_cv_scaled, y_train_cv)

    y_pred_proba_cv = model.predict_proba(X_val_cv_scaled)[:, 1]

    auc_score = roc_auc_score(y_val_cv, y_pred_proba_cv)
    auc_scores.append(auc_score)

    precision, recall, _ = precision_recall_curve(y_val_cv, y_pred_proba_cv)
    auc_pr_score = auc(recall, precision)
    auc_pr_scores.append(auc_pr_score)

    precision, recall, fscore, _ = precision_recall_fscore_support(y_val_cv, y_pred_proba_cv > 0.5)
    precision_scores.append(precision[1])  
    recall_scores.append(recall[1])  
    fscore_scores.append(fscore[1])  

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

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model.fit(X_scaled, y)
joblib.dump(model, "logistic_regression_model.pkl")
print("Model saved to logistic_regression_model.pkl")

y_pred = model.predict(X_scaled)
y_pred_proba = model.predict_proba(X_scaled)[:, 1]

print("Classification Report:")
print(classification_report(y, y_pred))

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

roc_auc = roc_auc_score(y, y_pred_proba)
print(f"AUC-ROC on full dataset: {roc_auc:.4f}")
