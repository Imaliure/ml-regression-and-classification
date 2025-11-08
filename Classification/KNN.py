# =====================================================
# K-NEAREST NEIGHBORS (KNN) CLASSIFIER – EXPLANATION
# -----------------------------------------------------
# KNN is a supervised learning algorithm used for 
# classification and regression tasks.
#
# How It Works:
#   • Does not build a model; it is a lazy learner
#   • For a new instance, it finds the 'K' closest 
#     data points (neighbors) using distance metrics 
#     such as Euclidean distance
#   • The class with the majority vote among neighbors 
#     determines the prediction
#
# Key Points:
#   ✓ Simple and easy to implement
#   ✓ Sensitive to scale → Feature scaling is recommended
#   ✓ Performance depends on a good choice of 'K'
#   ✓ Works best on smaller datasets (inefficient for large ones)
#
# Objective of this script:
#   • Train a baseline KNN classifier
#   • Tune 'n_neighbors' using GridSearchCV
#   • Evaluate performance before and after tuning
# =====================================================

# ================================
# Import Required Libraries
# ================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

# ================================
# Load Dataset
# ================================
df = pd.read_csv("../Data/diabetes.csv")

# Features & Target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# ================================
# Split into Train & Test
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ================================
# Baseline KNN Model (Default k=5)
# ================================
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Probability of class 1 (diabetes positive)
y_pred_proba = knn.predict_proba(X_test)[:, 1]

# Convert probabilities to class (0/1) using threshold 0.5
y_pred = y_pred_proba > 0.5

# Evaluation Metrics
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print("Baseline Accuracy:", accuracy)

# ================================
# Hyperparameter Tuning
# ================================
knn_params = {"n_neighbors": range(1, 20)}

grid = GridSearchCV(knn, knn_params, cv=10, verbose=2, n_jobs=-1)
grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)

# ================================
# Train Tuned KNN Model
# ================================
knn_tuned = KNeighborsClassifier(n_neighbors=grid.best_params_["n_neighbors"])
knn_tuned.fit(X_train, y_train)

# Predictions with tuned model
y_pred_tuned = knn_tuned.predict_proba(X_test)[:, 1]
accuracy_tuned = accuracy_score(y_test, y_pred_tuned > 0.5)

print("Tuned Accuracy:", accuracy_tuned)
