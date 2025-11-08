# ==============================================
# MULTI-LAYER PERCEPTRON (MLP) CLASSIFICATION
# ----------------------------------------------
# MLP is a type of Artificial Neural Network (ANN) used in supervised learning
# for both classification and regression problems.
#
# How it works:
#   - Consists of an input layer, one or more hidden layers, and an output layer.
#   - Each neuron performs a weighted sum + activation function (ReLU, tanh, logistic, etc.).
#   - Backpropagation is used to minimize the loss function by adjusting weights.
#
# Why use MLP?
#   ✓ Can model non-linear relationships.
#   ✓ Works well when data is standardized.
#   ✓ Flexible architecture (custom hidden layers & neurons).
#
# Key Hyperparameters:
#   • hidden_layer_sizes: Architecture of the network (neurons per layer).
#   • alpha: L2 regularization (prevents overfitting).
#   • activation: Function that introduces non-linearity.
#   • solver: Optimization algorithm (adam, sgd, lbfgs).
# ==============================================

# ================================
# Import Required Libraries
# ================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

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
# Feature Scaling (Very Important for ANN)
# ================================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ================================
# 1. Initial MLP Model (Default Parameters)
# ================================
mlp = MLPClassifier(random_state=42).fit(X_train, y_train)
y_pred = mlp.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Initial Accuracy Score: {accuracy}")

# ================================
# 2. Hyperparameter Tuning using GridSearchCV
# ================================
mlp_params = {
    "alpha": [0.1, 0.5, 1, 3],                        # Regularization strength
    "hidden_layer_sizes": [(10, 10), (100, 100), 
                           (100, 100, 100), (3, 5)]   # Network architecture
}

grid = GridSearchCV(
    estimator=mlp,
    param_grid=mlp_params,
    cv=10,
    verbose=2,
    n_jobs=-1
).fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)
# Example Output: {'alpha': 3, 'hidden_layer_sizes': (100, 100)}

# ================================
# 3. Train Final Tuned MLP Model
# ================================
mlp_tuned = MLPClassifier(
    alpha=grid.best_params_["alpha"],
    hidden_layer_sizes=grid.best_params_["hidden_layer_sizes"],
    random_state=42
).fit(X_train, y_train)

y_pred_tuned = mlp_tuned.predict(X_test)
accuracy_tuned = accuracy_score(y_test, y_pred_tuned)

print(f"Tuned Accuracy Score: {accuracy_tuned}")
