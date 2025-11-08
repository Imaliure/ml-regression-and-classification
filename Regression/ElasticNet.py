# ==============================================
# ELASTIC NET REGRESSION – EXPLANATION
# ----------------------------------------------
# Elastic Net is a regularized regression method that combines 
# both Lasso (L1) and Ridge (L2) penalties.
#
# Objective Function:
#   Minimize Σ(yᵢ - ŷᵢ)² + α * [l1_ratio * Σ|βⱼ|  +  (1 - l1_ratio) * Σ(βⱼ²)]
#
# Why Elastic Net?
#   - Lasso (L1) performs feature selection but may fail when variables are correlated.
#   - Ridge (L2) keeps all variables but doesn't eliminate any.
#   - ElasticNet combines both → handles multicollinearity + removes irrelevant features.
#
# Hyperparameters:
#   - alpha: Regularization strength
#   - l1_ratio: Balance between Lasso (1 → pure Lasso) and Ridge (0 → pure Ridge)
# ==============================================

# ================================
# Import Required Libraries
# ================================
import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ================================
# Load Dataset
# ================================
df = pd.read_csv("..\Data\Hitters.csv")
df.info()

# Handling missing values in the target variable
df["Salary"] = df["Salary"].fillna(df["Salary"].mean())

# One-hot encode categorical variables
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
# ElasticNet Model – Default Parameters
# ================================
elasticnet = ElasticNet()  # Default alpha=1.0, l1_ratio=0.5
model = elasticnet.fit(X_train, y_train)

print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

# Predictions and evaluation (RMSE & R2)
y_pred = model.predict(X_test)
print("Test RMSE (Default ElasticNet):", np.sqrt(mean_squared_error(y_test, y_pred)))
print("Test R2 Score:", r2_score(y_test, y_pred))

# ================================
# ElasticNet Cross-Validation (Hyperparameter Tuning)
# ================================
alphas = 10**np.linspace(10, -2, 100) * 0.5  # Search range for alpha
l1_ratio = [0.1, 0.3, 0.5, 0.8]               # Mix of L1 and L2

elastic_cv = ElasticNetCV(alphas=alphas, cv=10, l1_ratio=l1_ratio).fit(X_train, y_train)

# Best parameters
print("Optimal Alpha:", elastic_cv.alpha_)
print("Best l1_ratio:", elastic_cv.l1_ratio_)
print("Best Coefficients:", elastic_cv.coef_)

# ================================
# Final Tuned ElasticNet Model
# ================================
enet_tuned = ElasticNet(alpha=elastic_cv.alpha_, l1_ratio=elastic_cv.l1_ratio_).fit(X_train, y_train)

# Predictions with tuned model
y_pred = enet_tuned.predict(X_test)
print("Test RMSE (Tuned):", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score (Tuned):", r2_score(y_test, y_pred))
