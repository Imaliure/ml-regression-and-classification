import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split ,  GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier

df = pd.read_csv("..\Data\diabetes.csv")

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

cbc = CatBoostClassifier().fit(X_train, y_train)
y_pred = cbc.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

cbc_params = {
    "iterations": [200, 500],
    "learning_rate": [0.01, 0.1],
    "depth": [3, 6]
}

grid = GridSearchCV(cbc, cbc_params, cv=10, verbose=2, n_jobs=-1).fit(X_train, y_train)
grid.best_params_ # Out[16]: {'depth': 3, 'iterations': 200, 'learning_rate': 0.1}

cbc_tuned = CatBoostClassifier(iterations=200, learning_rate=0.1, depth=3).fit(X_train, y_train)
y_pred = cbc_tuned.predict(X_test)
accuracy_tuned = accuracy_score(y_test, y_pred)

print(f"Accuracy Score: {accuracy}")
print(f"Accuracy Score Tuned: {accuracy_tuned}")