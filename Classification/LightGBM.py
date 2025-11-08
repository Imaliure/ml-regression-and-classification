# =====================================================
# LIGHTGBM CLASSIFIER – GRADIENT BOOSTING BASED MODEL
# -----------------------------------------------------
# LightGBM (Light Gradient Boosting Machine) is a fast, 
# efficient, and scalable gradient boosting framework 
# developed by Microsoft.
#
# Why LightGBM?
#   ✓ Handles large datasets efficiently
#   ✓ Faster training using histogram-based algorithms
#   ✓ Supports categorical features and missing values
#   ✓ Often performs better than traditional models like RF or GBM
#
# Objective of this script:
#   • Train a baseline LightGBM classifier
#   • Tune hyperparameters using GridSearchCV
#   • Compare accuracy before and after tuning
# =====================================================

# ================================
# Import Libraries
# ================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier

# ================================
# Load Dataset
# ================================
df = pd.read_csv("../Data/diabetes.csv")

# Features & Target Variable
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# ================================
# Train-Test Split
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ================================
# Baseline LightGBM Model
# ================================
lgbmc = LGBMClassifier().fit(X_train, y_train)
y_pred = lgbmc.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Baseline Accuracy:", accuracy)

# ================================
# Hyperparameter Grid for Tuning
# ================================
lgbmc_params = {
    "learning_rate": [0.1, 0.5, 1],
    "n_estimators": [100, 200, 500, 1000],
    "max_depth": [3, 5, 7]
}

# ================================
# GridSearchCV – Hyperparameter Optimization
# ================================
grid = GridSearchCV(
    lgbmc, 
    lgbmc_params, 
    cv=10, 
    verbose=2, 
    n_jobs=-1
).fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)

# ================================
# Tuned LightGBM Model
# ================================
lgbmc_tuned = LGBMClassifier(
    learning_rate=grid.best_params_["learning_rate"],
    max_depth=grid.best_params_["max_depth"],
    n_estimators=grid.best_params_["n_estimators"]
).fit(X_train, y_train)

y_pred_tuned = lgbmc_tuned.predict(X_test)
accuracy_tuned = accuracy_score(y_test, y_pred_tuned)

print(f"Tuned Accuracy Score: {accuracy_tuned}")
