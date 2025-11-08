# ==============================================
# RIDGE REGRESSION – EXPLANATION
# ----------------------------------------------
# Ridge Regression is a regularized version of Linear Regression 
# that uses L2 regularization. It reduces overfitting by penalizing
# large coefficient values without making them exactly zero.
#
# Objective Function:
#   Minimize Σ(yᵢ - ŷᵢ)² + α * Σ(βⱼ²)
#
# Key Characteristics:
#   - Penalizes the sum of squared coefficients (L2 penalty)
#   - Keeps all variables in the model (does NOT set any coefficient to zero)
#   - Useful when multicollinearity exists
#   - α (alpha): Regularization strength. Higher α → more penalty → coefficients shrink more.
# ==============================================

# ================================
# Import Required Libraries
# ================================
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt

# ================================
# Load and Prepare Dataset
# ================================
df = pd.read_csv("Data\Hitters.csv")
df.info()

# Fill missing salary values with mean
df["Salary"] = df["Salary"].fillna(df["Salary"].mean())

# One-hot encoding for categorical variables
dms = pd.get_dummies(df[["League", "Division", "NewLeague"]])
X_ = df.drop(columns=["Salary", "League", "Division", "NewLeague"])
X = pd.concat([X_, dms[['League_N', 'Division_W', 'NewLeague_N']]], axis=1)
y = df["Salary"]

# ================================
# Train-Test Split
# ================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# ================================
# Initial Ridge Model (Manual alpha)
# ================================
ridge = Ridge(alpha=0.1)
model = ridge.fit(X_train, y_train)

print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

# Predictions & RMSE on test data
y_pred = model.predict(X_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))
print("Test RMSE (alpha=0.1):", rmse_test)

# ================================
# Ridge Paths: How Coefficients Change with Alpha
# ================================
alphas = 10**np.linspace(10, -2, 100) * 0.5
ridge_model = Ridge()
coefficients = []

for i in alphas:
    ridge_model.set_params(alpha=i)
    ridge_model.fit(X_train, y_train)
    coefficients.append(ridge_model.coef_)

ax = plt.gca()
ax.plot(alphas, coefficients)
ax.set_xscale("log")
ax.set_title("Ridge Coefficients vs Alpha")
ax.set_xlabel("Alpha (log scale)")
ax.set_ylabel("Coefficient Values")
plt.show()

# ================================
# Cross-Validation with Default Ridge
# ================================
ridge_model = Ridge().fit(X_train, y_train)
y_pred_train = ridge_model.predict(X_train)
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))

y_pred_test = ridge_model.predict(X_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
print("Default Ridge Test RMSE:", rmse_test)

# 10-fold cross-validation MSE
cv_score = cross_val_score(ridge_model, X_train, y_train, cv=10, scoring="neg_mean_squared_error")
print("Cross-Validated RMSE:", np.sqrt(np.mean(-cv_score)))

# ================================
# Best Alpha Selection using RidgeCV
# ================================
alphas_random = np.random.randint(0, 1000, 100)    # random alpha values
ridge_cv = RidgeCV(alphas=alphas_random, scoring="neg_mean_squared_error", cv=10)
ridge_cv.fit(X_train, y_train)

print("Best Alpha from RidgeCV:", ridge_cv.alpha_)

# ================================
# Train Final Ridge Model with the Best Alpha
# ================================
ridge_tuned = Ridge(alpha=ridge_cv.alpha_).fit(X_train, y_train)

# Train and test errors
y_pred_train = ridge_tuned.predict(X_train)
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))

y_pred_test = ridge_tuned.predict(X_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

print("Final Model Train RMSE:", rmse_train)
print("Final Model Test RMSE:", rmse_test)
