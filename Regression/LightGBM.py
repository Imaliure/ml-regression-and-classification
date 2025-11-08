# ==============================================
# LIGHTGBM REGRESSION – EXPLANATION
# ----------------------------------------------
# LightGBM (Light Gradient Boosting Machine) is a fast and efficient 
# gradient boosting algorithm developed by Microsoft.
#
# Key Advantages:
#   - Extremely fast training using histogram-based splitting
#   - Handles large datasets and high-dimensional data efficiently
#   - Supports missing values natively
#   - Uses leaf-wise tree growth instead of level-wise (faster and deeper trees)
#
# Important Parameters:
#   - learning_rate: Shrinks contribution of each tree to prevent overfitting
#   - n_estimators: Number of boosting trees
#   - max_depth: Maximum depth of each decision tree
#
# Objective:
#   Minimize prediction error by sequentially building decision trees 
#   that correct previous errors (Gradient Boosting logic)
# ==============================================

# ================================
# Import Required Libraries
# ================================
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt

# ================================
# Load and Prepare Dataset
# ================================
df = pd.read_csv("..\Data\Hitters.csv")
df.info()

# Handle missing values in target variable
df["Salary"] = df["Salary"].fillna(df["Salary"].mean())

# Encode categorical variables
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
# Initial LightGBM Model (Default)
# ================================
lightgbm_model = LGBMRegressor().fit(X_train, y_train)

# Predictions and evaluation
y_pred = lightgbm_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"LightGBM Model RMSE (Default): {rmse}")

# ================================
# Hyperparameter Tuning using GridSearchCV
# ================================
lightgbm_params = {
    'learning_rate': [0.01, 0.1, 0.5, 1],
    'n_estimators': [20, 40, 100, 200, 500, 1000],
    'max_depth': [1, 3, 5, 7, 9]
}

grid = GridSearchCV(
    lightgbm_model, lightgbm_params, cv=10, verbose=2, n_jobs=-1
).fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)

# Example Output:
# {'learning_rate': 0.1, 'max_depth': 1, 'n_estimators': 200}

# ================================
# Tuned LightGBM Model
# ================================
lightgbm_model_tuned = LGBMRegressor(
    learning_rate=0.1, 
    max_depth=1, 
    n_estimators=200
).fit(X_train, y_train)

# Evaluate tuned model
y_pred = lightgbm_model_tuned.predict(X_test)
rmse_tuned = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Tuned LightGBM RMSE: {rmse_tuned}")
