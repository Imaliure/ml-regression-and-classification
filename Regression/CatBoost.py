# ==============================================
# CATBOOST REGRESSION – EXPLANATION (ENGLISH)
# ----------------------------------------------
# CatBoost (Categorical Boosting) is a gradient boosting algorithm
# developed by Yandex. It is designed to work exceptionally well 
# with both numerical and categorical data.
#
# Key Advantages:
#   - Works well with categorical features (no need for one-hot encoding).
#   - Reduces overfitting using Ordered Boosting.
#   - Fast and accurate compared to many other boosting algorithms.
#
# Model Logic:
#   - Builds many decision trees sequentially.
#   - Each new tree tries to correct the errors of the previous ones.
#   - Final prediction = sum of predictions of all trees.
#
# Important Hyperparameters:
#   - iterations: Number of trees.
#   - learning_rate: How much each tree contributes to overall prediction.
#   - depth: Maximum depth of each tree.
# ==============================================

# ================================
# Import Libraries
# ================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor

# ================================
# Load Dataset
# ================================
df = pd.read_csv("Data\Hitters.csv")
df.info()

# Handle missing values
df["Salary"] = df["Salary"].fillna(df["Salary"].mean())

# One-hot encode categorical features (kept for consistency across models)
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
# Initial CatBoost Model
# ================================
catboost_model = CatBoostRegressor(verbose=False).fit(X_train, y_train)

# Predictions and error
y_pred = catboost_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"CatBoost RMSE (Default): {rmse}")

# ================================
# Hyperparameter Tuning - GridSearchCV
# ================================
catboost_params = {
    'iterations': [200, 500, 1000],
    'learning_rate': [0.01, 0.1],
    'depth': [3, 6]
}

grid = GridSearchCV(
    catboost_model,
    catboost_params,
    cv=10,
    verbose=2,
    n_jobs=-1
).fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)

# Example optimal output:
# {'depth': 6, 'iterations': 500, 'learning_rate': 0.01}

# ================================
# Train Tuned Model
# ================================
catboost_model_tuned = CatBoostRegressor(
    iterations=500,
    learning_rate=0.01,
    depth=6,
    verbose=False
).fit(X_train, y_train)

y_pred = catboost_model_tuned.predict(X_test)
rmse_tuned = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"CatBoost RMSE (Tuned): {rmse_tuned}")
