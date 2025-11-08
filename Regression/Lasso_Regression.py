# ==============================================
# LASSO REGRESSION – EXPLANATION
# ----------------------------------------------
# Lasso Regression (Least Absolute Shrinkage and Selection Operator) 
# is a linear regression model with L1 regularization.
# It shrinks less important feature coefficients to zero and performs 
# both variable selection and regularization.
#
# Mathematical Objective Function:
#   Minimize  Σ(yᵢ - ŷᵢ)² + α * Σ|βⱼ|
#
# Key Benefits:
#   - Performs feature selection by setting irrelevant coefficients to zero
#   - Reduces overfitting by penalizing large coefficients
#   - Useful when we have many correlated features
#
# Important Hyperparameter:
#   - alpha (λ): The penalty strength. 
#       Higher alpha → more regularization → more coefficients become zero.
# ==============================================

# ================================
# Import Required Libraries
# ================================
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso, LassoCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ================================
# Load and Prepare Dataset
# ================================
df = pd.read_csv("Data\Hitters.csv")
df.info()

# Fill missing values in target variable
df["Salary"] = df["Salary"].fillna(df["Salary"].mean())

# One-hot encoding for categorical features
dms = pd.get_dummies(df[["League", "Division", "NewLeague"]])

# Define feature matrix (X) and target vector (y)
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
# Initial Lasso Model (Default alpha)
# ================================
lasso = Lasso()
model = lasso.fit(X_train, y_train)

# Model coefficients (w values) and bias (b0)
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

# Predictions and error evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Test RMSE (Default Lasso):", np.sqrt(mse))
print("Test R2 Score:", r2_score(y_test, y_pred))

# ================================
# LassoCV - Hyperparameter Tuning (Best Alpha)
# ================================
# Define a range of alpha values (logarithmic scale)
alphas = 10**np.linspace(10, -2, 100) * 0.5

# Cross-validation with 10 folds to find best alpha
lasso_cv_model = LassoCV(alphas=alphas, cv=10, max_iter=100000).fit(X_train, y_train)
print("Optimal Alpha Found:", lasso_cv_model.alpha_)

# ================================
# Train Tuned Lasso Model
# ================================
lasso_tuned = Lasso(alpha=lasso_cv_model.alpha_).fit(X_train, y_train)

# Predict and evaluate tuned model
y_pred = lasso_tuned.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Test RMSE (Tuned Lasso):", np.sqrt(mse))
print("R2 Score (Tuned):", r2_score(y_test, y_pred))

# ================================
# Feature Importance (Non-zero coefficients)
# ================================
feature_importance = pd.Series(lasso_tuned.coef_, index=X_train.columns)
print(feature_importance)
