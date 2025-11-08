# ==============================================
# LINEAR REGRESSION – EXPLANATION
# ----------------------------------------------
# Linear Regression is a supervised learning algorithm 
# used to predict a continuous target (numeric value).
# It assumes a linear relationship between input features (X) 
# and the output variable (y).
#
# Mathematical Formula:
#     y = b0 + b1*x1 + b2*x2 + ... + bn*xn
#
# Objective:
#     To find the best-fitting line by minimizing 
#     the sum of squared errors (SSE) between actual 
#     and predicted values.
#
# Key Terms:
#     - b0: Intercept (bias)
#     - b1...bn: Coefficients (weights of features)
#     - y: Target / predicted value
#     - ε: Error term (difference between actual and predicted)
#
# Why do we use it?
#     - Simple and easy to interpret
#     - Works well when relationships are linear
#     - Baseline model for regression problems
# ==============================================

# ================================
# Import Required Libraries
# ================================
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error

# ================================
# Load Dataset
# ================================
df = pd.read_csv("Data/Advertising.csv")

# Display first few rows of the data
df.head()

# Show dataset information (column types, missing values, etc.)
df.info()

# ================================
# Data Visualization - Relationship Between TV and Sales
# ================================
# Jointplot shows regression line + scatter distribution
sns.jointplot(x="TV", y="Sales", data=df, kind="reg")

# Simple regression plot with red scatter points
sns.regplot(x=df["TV"], y=df["Sales"], ci=None, scatter_kws={'color': 'r', 's': 9})

# ================================
# Simple Linear Regression (Using only 'TV' as predictor)
# ================================
X = df[["TV"]]        # Independent variable
y = df[["Sales"]]     # Dependent variable

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Linear Regression model
lr = LinearRegression()
model = lr.fit(X_train, y_train)

# Display intercept (b0) and coefficient (b1)
print("Intercept:", model.intercept_)
print("Coefficient:", model.coef_)

# R-squared score (model performance on whole dataset)
print("R² Score:", model.score(X, y))

# Predict using test data
y_pred = model.predict(X_test)

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print("Test MSE:", mse)

# ================================
# Multiple Linear Regression (TV + Radio + Newspaper)
# Using Statsmodels for detailed statistical summary
# ================================
X = df.drop("Sales", axis=1)   # All independent variables
y = df["Sales"]                # Target variable

# Train-test split again for multi-variable model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit OLS (Ordinary Least Squares) model
lm = sm.OLS(y_train, X_train)
model = lm.fit()

# Display statistical model summary (p-values, R², F-test, etc.)
model.summary()

# Predict using statsmodels model
y_pred = model.predict(X_test)

# MSE for statsmodels predictions
mse = mean_squared_error(y_test, y_pred)
print("Statsmodels Test MSE:", mse)

# ================================
# Multiple Linear Regression with Scikit-Learn
# ================================
lm = LinearRegression()
model = lm.fit(X_train, y_train)

# Display model parameters
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

# Predictions for train and test sets
y_pred_test = model.predict(X_test)
y_pred_train = model.predict(X_train)

# Compute MSE for train and test
mse_test = mean_squared_error(y_test, y_pred_test)
mse_train = mean_squared_error(y_train, y_pred_train)

print("Test MSE:", mse_test)
print("Train MSE:", mse_train)

# ================================
# Cross-Validation (10-Fold CV)
# ================================
# Negative MSE values returned, so multiply by -1 to get positive MSE
cv_mse = -cross_val_score(model, X_train, y_train, cv=10, scoring="neg_mean_squared_error")
print("Cross-Validation MSE Scores:", cv_mse)
print("Mean CV MSE:", np.mean(cv_mse))

































