# =====================================================
# MODEL COMPARISON FOR CLASSIFICATION – EXPLANATION
# -----------------------------------------------------
# This script compares multiple machine learning classification models
# on the same dataset (Pima Indian Diabetes dataset).
#
# Purpose:
#   ✓ Train different classification models
#   ✓ Evaluate each model using Accuracy Score
#   ✓ Return results in a well-structured DataFrame
#
# Models Compared:
#   • Random Forest
#   • Decision Tree
#   • Support Vector Machine (SVM)
#   • K-Nearest Neighbors (KNN)
#   • Logistic Regression
#   • CatBoost
#   • XGBoost
#   • LightGBM
#   • Multi-Layer Perceptron (Neural Network)
#   • Gradient Boosting Classifier
#
# Dataset:
#   Pima Indians Diabetes (Outcome = 1 if diabetic, 0 otherwise)
#
# Output:
#   A table that ranks models based on accuracy on the test set.
# =====================================================

# ================================
# Import Required Libraries
# ================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Machine Learning Models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# ================================
# Load Dataset
# ================================
df = pd.read_csv("../Data/diabetes.csv")

# ================================
# Define Models to Compare
# ================================
models = [
    ("RandomForest", RandomForestClassifier(random_state=42)),
    ("DecisionTree", DecisionTreeClassifier(random_state=42)),
    ("SVM", SVC(probability=True, random_state=42)),
    ("KNN", KNeighborsClassifier()),
    ("LogisticRegression", LogisticRegression(max_iter=1000, solver="liblinear")),
    ("CatBoost", CatBoostClassifier(verbose=0, random_state=42)),
    ("XGBoost", XGBClassifier(eval_metric="logloss", random_state=42)),
    ("LightGBM", LGBMClassifier(random_state=42)),
    ("MLP", MLPClassifier(max_iter=500, random_state=42)),
    ("GradientBoosting", GradientBoostingClassifier(random_state=42))
]

# ================================
# Comparison Function
# ================================
def compML(df, target):
    """
    Compares multiple ML models on the same dataset using accuracy score.
    
    Parameters:
        df (DataFrame): Dataset containing features and target
        target (str): Target column name
        
    Returns:
        DataFrame: Model names and corresponding accuracy scores (sorted)
    """
    results = []
    X = df.drop(target, axis=1)
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    for name, model in models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results.append([name, accuracy])

    return pd.DataFrame(results, columns=["Model", "Accuracy"]).sort_values(
        by="Accuracy", ascending=False
    )

# ================================
# Run Model Comparison
# ================================
results_df = compML(df, "Outcome")
print(results_df)
