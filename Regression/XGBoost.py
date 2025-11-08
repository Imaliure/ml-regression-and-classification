# ==============================================
# XGBOOST REGRESSION – EXPLANATION
# ----------------------------------------------
# XGBoost (Extreme Gradient Boosting) is a powerful 
# and scalable tree-based ensemble learning algorithm.
#
# It works by building many decision trees sequentially, 
# where each new tree tries to correct the errors of the previous ones.
#
# Key Features:
#   - Gradient Boosting based decision tree model
#   - Regularization (L1 & L2) to reduce overfitting
#   - Handles missing values and non-linear relationships well
#   - Very fast and efficient due to parallelization and optimization
#
# Objective:
#   Minimize loss function using gradient descent in a stage-wise manner.
#
# Important Parameters:
#   - learning_rate: Shrinks contribution of each tree
#   - max_depth: Maximum depth of individual trees
#   - n_estimators: Number of boosting rounds (trees)
#   - subsample: Row sampling to prevent overfitting
# ==============================================

# ================================
# Import Required Libraries
# ================================
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

# ================================
# Load and Prepare Dataset
# ================================
df = pd.read_csv("Data\Hitters.csv")
df.info()

# Fill missing values in Salary column
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
# Initial XGBoost Model (Default Parameters)
# ================================
xgb_model = XGBRegressor().fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Default XGBoost RMSE:", rmse)

# ================================
# Hyperparameter Tuning (GridSearchCV)
# ================================
xgb_params = {
    "learning_rate": [0.1, 0.01, 0.5],
    "max_depth": [2, 3, 4, 5],
    "n_estimators": [100, 200, 500, 1000],
    "subsample": [0.6, 0.8, 1]
}

grid = GridSearchCV(xgb_model, xgb_params, cv=10, verbose=2, n_jobs=-1).fit(X_train, y_train)
print("Best Parameters:", grid.best_params_)

# Example Output:
# {'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 200, 'subsample': 0.6}

# ================================
# Train Tuned XGBoost Model
# ================================
xgb_model_tuned = XGBRegressor(
    learning_rate=0.01,
    max_depth=3,
    n_estimators=200,
    subsample=0.6
).fit(X_train, y_train)

# Predict and evaluate tuned model
y_pred_tuned = xgb_model_tuned.predict(X_test)
rmse_tuned = np.sqrt(mean_squared_error(y_test, y_pred_tuned))

print(f"Tuned XGBoost RMSE: {rmse_tuned}")

# ================================
# Feature Importance
# ================================
feature_importance = pd.DataFrame({
    "Value": xgb_model_tuned.feature_importances_ * 100,
    "Feature": X_train.columns
}).sort_values("Value", ascending=False)

print(feature_importance)
