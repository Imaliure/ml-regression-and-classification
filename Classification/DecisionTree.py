# =====================================================
# DECISION TREE CLASSIFIER – EXPLANATION
# -----------------------------------------------------
# Decision Tree is a supervised learning algorithm used 
# for both classification and regression tasks.
#
# How It Works:
#   • Data is split using decision rules based on features
#   • Creates a tree-like structure of nodes and leaves
#   • Each node represents a condition (feature-based split)
#   • Each leaf node represents a final decision/class
#
# Advantages:
#   ✓ Easy to interpret and visualize
#   ✓ Works with both numerical and categorical data
#   ✓ No feature scaling required
#
# Disadvantages:
#   ✕ Prone to overfitting if hyperparameters are not tuned
#   ✕ Less stable — small changes in data can change the tree
#
# Objective of this script:
#   • Train a baseline Decision Tree Classifier
#   • Tune hyperparameters (max_depth, min_samples_split)
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
from sklearn.tree import DecisionTreeClassifier

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
# Baseline Decision Tree Model
# ================================
dt_model = DecisionTreeClassifier().fit(X_train, y_train)
y_pred = dt_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Baseline Accuracy: {accuracy}")

# ================================
# Hyperparameter Tuning
# ================================
dt_params = {
    "max_depth": [1, 3, 5, 8, 10],
    "min_samples_split": [2, 3, 5, 10, 20],
}

grid = GridSearchCV(
    estimator=dt_model,
    param_grid=dt_params,
    cv=10,
    verbose=2,
    n_jobs=-1
).fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)

# ================================
# Tuned Decision Tree Model
# ================================
dt_model_tuned = DecisionTreeClassifier(
    max_depth=grid.best_params_["max_depth"],
    min_samples_split=grid.best_params_["min_samples_split"]
).fit(X_train, y_train)

y_pred_tuned = dt_model_tuned.predict(X_test)
accuracy_tuned = accuracy_score(y_test, y_pred_tuned)

print(f"Tuned Accuracy: {accuracy_tuned}")
