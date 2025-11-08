# ==============================================
# XGBOOST CLASSIFICATION – EXPLANATION
# ----------------------------------------------
# XGBoost (Extreme Gradient Boosting) is a powerful and efficient 
# implementation of Gradient Boosting for classification and regression.
#
# How it works:
#   - Builds many decision trees sequentially.
#   - Each new tree corrects the errors of the previous trees.
#   - Uses gradient descent to minimize the loss function.
#
# Why is it popular?
#   ✓ Very fast and optimized (uses parallel computing).
#   ✓ Handles missing data automatically.
#   ✓ Reduces overfitting using regularization (L1 & L2).
#   ✓ Frequently wins Kaggle competitions.
#
# Key Hyperparameters:
#   - learning_rate: Shrinks the contribution of each tree.
#   - max_depth: Maximum depth of decision trees.
#   - n_estimators: Number of boosting rounds (trees).
# ==============================================

# ================================
# Import Required Libraries
# ================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# ================================
# Load Dataset
# ================================
df = pd.read_csv("..\Data\diabetes.csv")

# Features and Target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# ================================
# Train-Test Split
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ================================
# 1. Initial XGBoost Model (Default Parameters)
# ================================
xgb = XGBClassifier().fit(X_train, y_train)

y_pred = xgb.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Initial Accuracy Score: {accuracy}")

# ================================
# 2. Hyperparameter Tuning using GridSearchCV
# ================================
xgb_params = {
    "learning_rate": [0.1, 0.5, 1],
    "max_depth": [3, 5, 7],
    "n_estimators": [100, 200, 300]
}

grid = GridSearchCV(
    estimator=xgb,
    param_grid=xgb_params,
    cv=10,
    verbose=2,
    n_jobs=-1
).fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)

# Example output:
# {'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 300}

# ================================
# 3. Train Final Tuned Model
# ================================
xgb_tuned = XGBClassifier(
    learning_rate=0.1,
    max_depth=3,
    n_estimators=300
).fit(X_train, y_train)

y_pred_tuned = xgb_tuned.predict(X_test)
accuracy_tuned = accuracy_score(y_test, y_pred_tuned)

print(f"Tuned Accuracy Score: {accuracy_tuned}")
