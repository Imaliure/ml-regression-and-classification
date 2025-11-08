# ==============================================
# GRADIENT BOOSTING REGRESSION – EXPLANATION
# ----------------------------------------------
# Gradient Boosting is an ensemble learning technique that builds multiple
# decision trees sequentially, where each tree tries to correct the errors 
# of the previous one.
#
# Key Idea:
#   - Initial model predicts the mean of y
#   - Next trees are trained on residual errors (y - y_pred)
#   - Final prediction is the sum of all weak learner predictions
#
# Objective:
#   Minimize a loss function (usually MSE) using gradient descent on errors.
#
# Key Hyperparameters:
#   - learning_rate: Shrinks the impact of each tree (smaller = less overfitting, but slower)
#   - n_estimators: Number of boosting stages (trees)
#   - max_depth: Depth of each individual regression tree
#   - subsample: Fraction of samples used per tree (used for stochastic gradient boosting)
#
# Advantages:
#   - High accuracy
#   - Handles non-linear relationships
#
# Disadvantages:
#   - Can overfit if learning_rate and tree depth are not tuned
#   - Slower than Random Forest for very large data
# ==============================================

# ================================
# Import Required Libraries
# ================================
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt

# ================================
# Load and Prepare Dataset
# ================================
df = pd.read_csv("..\Data\Hitters.csv")
df.info()

# Fill missing values in Salary
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
# Initial GBM Model (Default Parameters)
# ================================
gbm_model = GradientBoostingRegressor().fit(X_train, y_train)

# Predictions and evaluation
y_pred = gbm_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Gradient Boosting RMSE (Default): {rmse}")

# ================================
# Hyperparameter Tuning using GridSearchCV
# ================================
gbm_params = {
    "learning_rate": [0.001, 0.01, 0.1],
    "max_depth": [3, 5, 7],
    "n_estimators": [100, 500, 1000],
    "subsample": [1, 0.5, 0.7]
}

grid = GridSearchCV(
    gbm_model, gbm_params, cv=10, verbose=2, n_jobs=-1
).fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)

# Example Output:
# {'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 500, 'subsample': 0.5}

# ================================
# Train Tuned GBM Model
# ================================
gbm_model_tuned = GradientBoostingRegressor(
    learning_rate=0.01,
    max_depth=3,
    n_estimators=500,
    subsample=0.5
).fit(X_train, y_train)

# Evaluate tuned model
y_pred = gbm_model_tuned.predict(X_test)
rmse_tuned = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Gradient Boosting RMSE (Tuned): {rmse_tuned}")
