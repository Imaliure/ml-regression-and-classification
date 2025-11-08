# ==============================================
# MODEL COMPARISON FUNCTION FOR REGRESSION MODELS
# ----------------------------------------------
# This function:
#   ✓ Loads the dataset (Hitters.csv)
#   ✓ Preprocesses data (missing values + one-hot encoding)
#   ✓ Splits into train/test sets
#   ✓ Trains the given model
#   ✓ Calculates RMSE and returns results
#   ✓ Supports any regression model from sklearn / xgboost / catboost etc.
# ==============================================

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Regression Models to Compare
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


df = pd.read_csv("Data\Hitters.csv")

# List of models to test
models = [
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    DecisionTreeRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor,
    XGBRegressor,
    LGBMRegressor,
    CatBoostRegressor,
    SVR,
    KNeighborsRegressor,
    MLPRegressor
]

def compML(df, target, alg_list=models):
    results = []  # Store model name and RMSE
    
    # Preprocess dataset once
    df["Salary"] = df["Salary"].fillna(df["Salary"].mean())
    dms = pd.get_dummies(df[["League", "Division", "NewLeague"]])
    X_ = df.drop(columns=["Salary", "League", "Division", "NewLeague"])
    X = pd.concat([X_, dms[['League_N', 'Division_W', 'NewLeague_N']]], axis=1)
    y = df[target]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    
    # Run each model
    for alg in alg_list:
        model_name = alg.__name__  # Clean model name
        try:
            model = alg().fit(X_train, y_train)
            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            results.append([model_name, rmse])
        except:
            results.append([model_name, "Error"])

    # Convert to DataFrame and sort by RMSE
    results_df = pd.DataFrame(results, columns=["Model", "RMSE"]).sort_values(by="RMSE")
    return results_df

# Run and display results
compML(df, "Salary")
