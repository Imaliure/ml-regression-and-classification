# ==============================================
# SUPPORT VECTOR MACHINE (SVM) CLASSIFICATION
# ----------------------------------------------
# SVM is a supervised learning algorithm used for both
# classification and regression tasks.
#
# Main idea:
#   - Finds the best separating hyperplane that maximizes 
#     the margin between classes.
#
# Key Concepts:
#   • Support Vectors: Data points closest to the hyperplane.
#   • Kernel Trick: Transforms non-linear data into higher 
#     dimensions to make it linearly separable.
#
# In this example:
#   - We use a linear kernel for classification.
#   - We perform hyperparameter tuning over 'C' values.
#
# Advantages:
#   ✓ Works well on small to medium datasets.
#   ✓ Effective in high-dimensional spaces.
#   ✓ Robust to overfitting when properly tuned.
#
# Disadvantages:
#   ✗ Not ideal for very large datasets.
#   ✗ Requires feature scaling for non-linear kernels.
# ==============================================

# ================================
# Import Libraries
# ================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC

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
# 1. Initial SVM Model (Linear Kernel)
# ================================
svc = SVC(kernel="linear", probability=False).fit(X_train, y_train)
y_pred = svc.predict(X_test)

# Performance metrics
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
print("Initial Accuracy Score:", accuracy)

# ================================
# 2. Hyperparameter Tuning using GridSearchCV
# ================================
svc_params = {
    "C": [0.1, 0.5, 1, 3]   # Regularization parameter
}

grid = GridSearchCV(
    estimator=svc,
    param_grid=svc_params,
    cv=10,
    n_jobs=-1,
    verbose=2
).fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)

# ================================
# 3. Train Tuned SVM Model
# ================================
svc_tuned = SVC(kernel="linear", C=grid.best_params_["C"])
svc_tuned.fit(X_train, y_train)

# Predictions & final accuracy
y_pred_tuned = svc_tuned.predict(X_test)
accuracy_tuned = accuracy_score(y_test, y_pred_tuned)

print("Accuracy Score (Tuned):", accuracy_tuned)
