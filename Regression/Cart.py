# ==============================================
# DECISION TREE REGRESSION (CART) – EXPLANATION
# ----------------------------------------------
# CART (Classification and Regression Trees) is a tree-based algorithm
# used for regression and classification problems.
#
# How it works (Regression context):
#   - Splits the dataset based on a feature that minimizes the MSE (Mean Squared Error).
#   - Creates a tree where each leaf node represents a numerical prediction.
#
# Key Hyperparameters:
#   - max_depth: Maximum depth of the tree (prevents overfitting).
#   - min_samples_split: Minimum number of samples required to split a node.
#   - min_samples_leaf: Minimum samples required in a leaf node.
#
# Advantages:
#   - Easy to interpret and visualize.
#   - Captures non-linear relationships.
# Disadvantages:
#   - Prone to overfitting without tuning.
# ==============================================

# ================================
# Import Required Libraries
# ================================
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

# ================================
# Load Dataset
# ================================
df = pd.read_csv("..\Data\Hitters.csv")
df.info()

# Handle missing values
df["Salary"] = df["Salary"].fillna(df["Salary"].mean())

# One-hot encoding for categorical variables
dms = pd.get_dummies(df[["League", "Division", "NewLeague"]])
X_ = df.drop(columns=["Salary", "League", "Division", "NewLeague"])
X = pd.concat([X_, dms[['League_N', 'Division_W', 'NewLeague_N']]], axis=1)
y = df["Salary"]

# ================================
# 1. CART Using Only 'Hits' Feature (Simple Tree)
# ================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Use only 'Hits' as a single predictor
X_train_hits = pd.DataFrame(X_train["Hits"])
X_test_hits = pd.DataFrame(X_test["Hits"])

cart_model = DecisionTreeRegressor()
cart_model.fit(X_train_hits, y_train)

# Plot the regression tree fit
X_grid = np.arange(min(X_train_hits["Hits"]), max(X_train_hits["Hits"]), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))

plt.scatter(X_train_hits, y_train, color="red")
plt.plot(X_grid, cart_model.predict(X_grid), color="blue")
plt.title("Decision Tree Regression (Using Hits Only)")
plt.xlabel("Hits")
plt.ylabel("Salary")
plt.show()

# RMSE with only Hits feature
y_pred = cart_model.predict(X_test_hits)
rmse_hits = np.sqrt(mean_squared_error(y_test, y_pred))

# ================================
# 2. CART Using All Features
# ================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

cart_model_all = DecisionTreeRegressor()
cart_model_all.fit(X_train, y_train)

y_pred_all = cart_model_all.predict(X_test)
rmse_all = np.sqrt(mean_squared_error(y_test, y_pred_all))

# ================================
# 3. Hyperparameter Tuning with GridSearchCV
# ================================
cart_params = {
    "max_depth": [1, 3, 5, 7, 9],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1, 5, 10, 20]
}

grid = GridSearchCV(cart_model_all, cart_params, cv=10, verbose=2, n_jobs=-1)
grid.fit(X_train, y_train)

print("Best Hyperparameters:", grid.best_params_)
# Example result:
# {'max_depth': 3, 'min_samples_leaf': 20, 'min_samples_split': 20}

# Train final tuned model
cart_tuned = DecisionTreeRegressor(
    max_depth=grid.best_params_["max_depth"],
    min_samples_leaf=grid.best_params_["min_samples_leaf"],
    min_samples_split=grid.best_params_["min_samples_split"]
).fit(X_train, y_train)

y_pred_tuned = cart_tuned.predict(X_test)
rmse_tuned = np.sqrt(mean_squared_error(y_test, y_pred_tuned))

# ================================
# Results
# ================================
print(f"RMSE (Only Hits): {rmse_hits}")
print(f"RMSE (All Features - Untuned): {rmse_all}")
print(f"RMSE (All Features - Tuned): {rmse_tuned}")
