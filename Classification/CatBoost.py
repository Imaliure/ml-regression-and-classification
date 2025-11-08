# =====================================================
# CATBOOST CLASSIFIER – EXPLANATION
# -----------------------------------------------------
# CatBoost (Categorical Boosting) is a gradient boosting 
# algorithm developed by Yandex. It is designed to work 
# efficiently with both numerical and categorical data.
#
# How It Works:
#   • Uses decision trees as weak learners in a boosting framework
#   • Handles categorical features automatically using target encoding
#   • Reduces overfitting using ordered boosting technique
#
# Advantages:
#   ✓ Handles categorical features without one-hot encoding
#   ✓ Fast training and robust to overfitting
#   ✓ Works well with small to medium-sized datasets
#
# Disadvantages:
#   ✕ Can be slower than Random Forest for large datasets
#   ✕ Requires CatBoost library installation (not in sklearn by default)
#
# Objective of this script:
#   • Train a baseline CatBoost Classifier
#   • Tune hyperparameters using GridSearchCV
#   • Compare accuracy before and after tuning
# =====================================================

# ================================
# Import Required Libraries
# ================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier

# ================================
# Load Dataset
# ================================
df = pd.read_csv("../Data/diabetes.csv")

# Define Features and Target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# ================================
# Split into Train and Test Sets
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ================================
# Baseline CatBoost Model
# ================================
cbc = CatBoostClassifier(verbose=0).fit(X_train, y_train)  # verbose=0 to suppress training output
y_pred = cbc.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Baseline Accuracy: {accuracy}")

# ================================
# Hyperparameter Tuning using GridSearchCV
# ================================
cbc_params = {
    "iterations": [200, 500],
    "learning_rate": [0.01, 0.1],
    "depth": [3, 6]
}

grid = GridSearchCV(
    estimator=cbc,
    param_grid=cbc_params,
    cv=10,
    verbose=2,
    n_jobs=-1
).fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)

# ================================
# Tuned CatBoost Model
# ================================
cbc_tuned = CatBoostClassifier(
    iterations=grid.best_params_["iterations"],
    learning_rate=grid.best_params_["learning_rate"],
    depth=grid.best_params_["depth"],
    verbose=0
).fit(X_train, y_train)

y_pred_tuned = cbc_tuned.predict(X_test)
accuracy_tuned = accuracy_score(y_test, y_pred_tuned)

print(f"Tuned Accuracy: {accuracy_tuned}")
