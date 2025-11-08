# =====================================================
# GRADIENT BOOSTING CLASSIFIER – EXPLANATION
# -----------------------------------------------------
# Gradient Boosting is an ensemble learning technique 
# used for classification and regression tasks.
#
# How it Works:
#   • Builds models sequentially (tree by tree)
#   • Each new model tries to correct the errors of 
#     the previous models by minimizing the loss function
#   • Final prediction = weighted sum of all weak learners
#
# Advantages:
#   ✓ Handles non-linear relationships
#   ✓ Works well with complex datasets
#   ✓ High predictive power
#
# Disadvantages:
#   ✕ Can overfit without tuning
#   ✕ Slower than Random Forest due to sequential training
#
# Objective of this script:
#   • Train a baseline Gradient Boosting model
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
from sklearn.ensemble import GradientBoostingClassifier

# ================================
# Load Dataset
# ================================
df = pd.read_csv("../Data/diabetes.csv")

# Define Features and Target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# ================================
# Split Dataset (Train/Test)
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ================================
# Baseline Gradient Boosting Model
# ================================
gbm = GradientBoostingClassifier().fit(X_train, y_train)

# Predictions
y_pred = gbm.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Baseline Accuracy: {accuracy}")

# ================================
# Hyperparameter Tuning with GridSearchCV
# ================================
gbm_params = {
    "learning_rate": [0.001, 0.01, 0.1, 0.5],
    "n_estimators": [100, 500, 1000],
    "max_depth": [3, 5, 8]
}

grid = GridSearchCV(
    gbm, 
    gbm_params, 
    cv=10, 
    verbose=2, 
    n_jobs=-1
).fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)

# ================================
# Tuned Model with Best Parameters
# ================================
gbm_tuned = GradientBoostingClassifier(
    learning_rate=grid.best_params_["learning_rate"],
    max_depth=grid.best_params_["max_depth"],
    n_estimators=grid.best_params_["n_estimators"]
).fit(X_train, y_train)

# Predictions and Accuracy
y_pred_tuned = gbm_tuned.predict(X_test)
accuracy_tuned = accuracy_score(y_test, y_pred_tuned)

print(f"Tuned Accuracy: {accuracy_tuned}")
