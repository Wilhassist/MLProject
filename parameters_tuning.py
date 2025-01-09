#!/usr/bin/env python3
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, precision_recall_curve, auc
from sklearn.preprocessing import StandardScaler

from data_preprocessing.resampling_data import data_oversampled
data = data_oversampled()

X = data.drop(columns=['label'])
y = data['label']

# Custom scoring function for AUC-PR
def auc_pr_score(y_true, y_pred_proba):
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    return auc(recall, precision)

scorer = make_scorer(auc_pr_score, needs_proba=True)

# Define the parameter grid
param_dist = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

# Initialize the model
rf_model = RandomForestClassifier(class_weight='balanced', random_state=42)

# RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=rf_model,
    param_distributions=param_dist,
    n_iter=50,  # Number of combinations to try
    scoring=scorer,
    cv=5,
    verbose=2,
    random_state=42,
    n_jobs=-1  # Use all available cores
)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform the search
random_search.fit(X_scaled, y)

# Best parameters and score
print(f"Best Parameters: {random_search.best_params_}")
print(f"Best AUC-PR Score: {random_search.best_score_:.4f}")

# Train the model with the best parameters on the entire dataset
best_rf_model = random_search.best_estimator_
best_rf_model.fit(X_scaled, y)

# Evaluate the tuned model
y_pred_proba = best_rf_model.predict_proba(X_scaled)[:, 1]
precision, recall, _ = precision_recall_curve(y, y_pred_proba)
final_auc_pr = auc(recall, precision)

print(f"Final AUC-PR on the full dataset: {final_auc_pr:.4f}")

# Plot Precision-Recall Curve
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f"AUC-PR = {final_auc_pr:.4f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve (Tuned Model)")
plt.legend()
plt.grid()
plt.show()
