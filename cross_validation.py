import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_recall_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np

# Load the balanced dataset
positive_data = pd.read_csv("../data/training_pos_features.csv", index_col=0)
unlabelled_data = pd.read_csv("../data/training_others_features.csv", index_col=0)

positive_data['label'] = 1
unlabelled_data['label'] = 0

data = pd.concat([positive_data, unlabelled_data], axis=0)
data = data.sample(frac=1, random_state=42)

# Split the dataset into features and labels
X = data.drop(columns=["label"])  
y = data["label"]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize the model
model = LogisticRegression(class_weight="balanced", random_state=42)

# Perform 5-fold cross-validation
cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"Cross-validation accuracy scores: {cv_scores}")
print(f"Mean accuracy from cross-validation: {np.mean(cv_scores)}")

# Train the model on the full dataset
model.fit(X_scaled, y)

# Save the trained model
joblib.dump(model, "logistic_regression_model.pkl")
print("Model saved to logistic_regression_model.pkl")

# Make predictions
y_pred = model.predict(X_scaled)
y_pred_proba = model.predict_proba(X_scaled)[:, 1]

# Evaluate the model using the classification report
print("Classification Report:")
print(classification_report(y, y_pred))

# Calculate AUC-PR
precision, recall, thresholds = precision_recall_curve(y, y_pred_proba)
auc_pr = auc(recall, precision)
print(f"AUC-PR: {auc_pr:.4f}")

# Plot Precision-Recall Curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f"AUC-PR = {auc_pr:.4f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.grid()
plt.show()

# Evaluate the model on the AUC-ROC (using StratifiedKFold cross-validation)
skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
auc_scores = []
for train_index, val_index in skf.split(X, y):
    X_train_cv, X_val_cv = X.iloc[train_index], X.iloc[val_index]
    y_train_cv, y_val_cv = y.iloc[train_index], y.iloc[val_index]
    
    X_train_cv_scaled = scaler.fit_transform(X_train_cv)
    X_val_cv_scaled = scaler.transform(X_val_cv)
    
    model.fit(X_train_cv_scaled, y_train_cv)
    y_pred_proba_cv = model.predict_proba(X_val_cv_scaled)[:, 1]
    auc_score = roc_auc_score(y_val_cv, y_pred_proba_cv)
    auc_scores.append(auc_score)

print(f"AUC-ROC scores for each fold: {auc_scores}")
print(f"Mean AUC-ROC from cross-validation: {np.mean(auc_scores)}")
