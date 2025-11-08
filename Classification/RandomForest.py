# ==============================================
# RANDOM FOREST CLASSIFICATION – EXPLANATION
# ----------------------------------------------
# Random Forest is an ensemble learning algorithm that builds multiple
# decision trees and combines their results to improve accuracy and reduce overfitting.
#
# How it works:
#   ✓ Bootstrapping: Each tree is trained on a random subset of the dataset.
#   ✓ Feature Randomness: At each split, only a random subset of features is considered.
#   ✓ Final Prediction: Majority vote (classification) or average (regression).
#
# Advantages:
#   ✓ Handles both classification and regression tasks.
#   ✓ Robust to noise and outliers.
#   ✓ Reduces overfitting compared to a single Decision Tree.
#
# Key Hyperparameters:
#   • n_estimators: Number of trees in the forest.
#   • max_depth: Maximum depth of each tree.
#   • max_features: Number of features considered for splitting.
# ==============================================

# ================================
# Import Libraries
# ================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# ================================
# Load Dataset
# ================================
df = pd.read_csv("..\Data\diabetes.csv")

# Feature matrix (X) and target vector (y)
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# ================================
# Train-Test Split
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ================================
# 1. Initial Random Forest Model (Default Parameters)
# ================================
rf_model = RandomForestClassifier(random_state=42).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Initial Accuracy Score: {accuracy}")

# ================================
# 2. Hyperparameter Tuning using GridSearchCV
# ================================
rf_params = {
    "max_depth": [1, 3, 5, 8, 10],
    "n_estimators": [10, 50, 100, 500]
}

grid = GridSearchCV(
    estimator=rf_model,
    param_grid=rf_params,
    cv=10,
    verbose=2,
    n_jobs=-1
).fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)
# Example Output → {'max_depth': 8, 'n_estimators': 500}

# ================================
# 3. Train Final Tuned RF Model
# ================================
rf_model_tuned = RandomForestClassifier(
    max_depth=grid.best_params_["max_depth"],
    n_estimators=grid.best_params_["n_estimators"],
    random_state=42
).fit(X_train, y_train)

y_pred_tuned = rf_model_tuned.predict(X_test)
accuracy_tuned = accuracy_score(y_test, y_pred_tuned)

print(f"Tuned Accuracy Score: {accuracy_tuned}")
