# ==============================================
# K-NEAREST NEIGHBORS REGRESSION – EXPLANATION
# ----------------------------------------------
# KNN (K-Nearest Neighbors) Regression is a non-parametric 
# machine learning method used to predict continuous values.
#
# How it works:
#   - For a new data point, the algorithm finds the 'k' closest 
#     training data points (neighbors) using a distance metric (default: Euclidean).
#   - The prediction is computed as the average of target values from these neighbors.
#
# Key Hyperparameter:
#   - n_neighbors (k): Number of neighbors to consider.
#       Small k → High variance, overfitting.
#       Large k → High bias, underfitting.
#
# Advantages:
#   - Simple and intuitive algorithm.
#   - No training phase (lazy learner).
#
# Disadvantages:
#   - Slow for large datasets during prediction.
#   - Sensitive to feature scaling → Standardization may be required.
# ==============================================

# ================================
# Import Required Libraries
# ================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from warnings import filterwarnings
filterwarnings("ignore")

# ================================
# Load and Prepare Dataset
# ================================
df = pd.read_csv("..\Data\Hitters.csv")
df.info()

# Fill missing Salary values
df["Salary"] = df["Salary"].fillna(df["Salary"].mean())

# One-hot encode categorical features
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
# Initial KNN Model (Default k)
# ================================
knn = KNeighborsRegressor().fit(X_train, y_train)
y_pred = knn.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Initial KNN RMSE:", rmse)

# ================================
# Try Different k Values (Manual Search)
# ================================
RMSE = []
for k in range(1, 11):
    knn = KNeighborsRegressor(n_neighbors=k).fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    rmse_k = np.sqrt(mean_squared_error(y_test, y_pred))
    RMSE.append(rmse_k)
    print("k =", k, "RMSE =", rmse_k)

print("Minimum RMSE:", min(RMSE), "| Maximum RMSE:", max(RMSE))

# ================================
# Hyperparameter Tuning using GridSearchCV
# ================================
knn_params = {"n_neighbors": np.arange(1, 30, 1)}

grid = GridSearchCV(KNeighborsRegressor(), knn_params, cv=10).fit(X_train, y_train)
print("Best Parameters from GridSearch:", grid.best_params_)

# Final Prediction using best k
y_pred = grid.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Final KNN RMSE (Tuned):", rmse)
