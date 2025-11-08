# ==============================================
# RANDOM FOREST REGRESSION – EXPLANATION
# ----------------------------------------------
# Random Forest is an ensemble learning method that builds multiple 
# decision trees and combines their results to improve prediction accuracy.
#
# How it works:
#   - Trains many decision trees on different random subsets of data (Bootstrap sampling)
#   - Each tree votes (regression: averages predictions)
#   - Reduces overfitting and improves generalization compared to a single decision tree
#
# Key Parameters:
#   - n_estimators: Number of trees in the forest
#   - max_depth: Maximum depth of each tree
#   - max_features: Number of features to consider at each split
#   - min_samples_split: Minimum samples needed to split a node
#
# Advantages:
#   - Handles non-linear relationships & high-dimensional data
#   - Performs feature importance analysis
#   - Less overfitting than a single decision tree
# ==============================================

# ================================
# Import Required Libraries
# ================================
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# ================================
# Load and Prepare Dataset
# ================================
df = pd.read_csv("Data\Hitters.csv")
df.info()

# Fill missing target values
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
# Initial Random Forest Model (Default Parameters)
# ================================
rf_model = RandomForestRegressor(random_state=42).fit(X_train, y_train)

# Predictions and RMSE
y_pred = rf_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Random Forest Model RMSE (Default): {rmse}")

# ================================
# Hyperparameter Tuning using GridSearchCV
# ================================
rf_params = {
    "max_depth": range(1, 10),
    "max_features": [2, 3, 4],
    "n_estimators": [100, 200, 500, 1000],
    "min_samples_split": [2, 5, 10]
}

grid = GridSearchCV(
    rf_model, rf_params, cv=10, verbose=2, n_jobs=-1
).fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)

# Example Output:
# {'max_depth': 9, 'max_features': 2, 'min_samples_split': 2, 'n_estimators': 200}

# ================================
# Tuned Random Forest Model
# ================================
rf_model_tuned = RandomForestRegressor(
    max_depth=9,
    max_features=2,
    min_samples_split=2,
    n_estimators=200,
    random_state=42
).fit(X_train, y_train)

# Predictions and RMSE after tuning
y_pred_tuned = rf_model_tuned.predict(X_test)
rmse_tuned = np.sqrt(mean_squared_error(y_test, y_pred_tuned))

print(f"Tuned Random Forest RMSE: {rmse_tuned}")

# ================================
# Feature Importance Visualization (Optional)
# ================================
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model_tuned.feature_importances_ * 100
}).sort_values("Importance", ascending=False)

print(feature_importance)
