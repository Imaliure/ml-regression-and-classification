# ==============================================
# SUPPORT VECTOR REGRESSION (SVR) – EXPLANATION
# ----------------------------------------------
# SVR is the regression version of Support Vector Machines (SVM).
# It aims to find a function that fits the data with the smallest error 
# within a certain margin (epsilon-insensitive tube).
#
# Objective:
#   - Fit a curve/line that keeps most data points within ε distance.
#   - Minimize model complexity and prediction error simultaneously.
#
# Key Parameters:
#   - kernel: Defines the transformation (linear, poly, rbf, sigmoid)
#   - C: Regularization parameter (high C → low bias, risk of overfitting)
#   - epsilon: Margin of tolerance where errors are ignored
#
# Why use SVR?
#   - Works well for non-linear relationships
#   - Uses kernel trick for high-dimensional feature mapping
#   - Robust against outliers when epsilon is well-tuned
# ==============================================

# ================================
# Import Required Libraries
# ================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from warnings import filterwarnings
filterwarnings("ignore")

# ================================
# Load and Prepare Dataset
# ================================
df = pd.read_csv("Data\Hitters.csv")
df.info()

# Handle missing Salary values
df["Salary"] = df["Salary"].fillna(df["Salary"].mean())

# One-hot encoding for categorical columns
dms = pd.get_dummies(df[["League", "Division", "NewLeague"]])
X_ = df.drop(columns=["Salary", "League", "Division", "NewLeague"])
X = pd.concat([X_, dms[['League_N', 'Division_W', 'NewLeague_N']]], axis=1)
y = df["Salary"]

# ================================
# Train-Test Split
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# ================================
# Initial SVR Model (Linear Kernel)
# ================================
svr = SVR(kernel="linear").fit(X_train, y_train)

# Model parameters (weights & intercept for linear kernel)
print("Intercept:", svr.intercept_)
print("Coefficients:", svr.coef_)

# Predictions and RMSE
y_pred = svr.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Default SVR RMSE:", rmse)

# ================================
# Hyperparameter Tuning using GridSearchCV
# ================================
svr_params = {
    "kernel": ['linear', 'poly', 'rbf', 'sigmoid'],
    "epsilon": [0.01, 0.1, 1, 10],
    "C": [0.1, 0.5, 1, 3]
}

grid = GridSearchCV(svr, svr_params, cv=10, verbose=2, n_jobs=-1).fit(X_train, y_train)
print("Best Parameters:", grid.best_params_)

# ================================
# Final Model with Best Parameters
# ================================
best_svr = grid.best_estimator_
y_pred = best_svr.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Tuned SVR RMSE:", rmse)
print("Tuned SVR R2 Score:", r2)
