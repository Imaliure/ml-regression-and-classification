import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split ,  GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier

df = pd.read_csv("..\Data\diabetes.csv")

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lgbmc = LGBMClassifier().fit(X_train, y_train)
y_pred = lgbmc.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

lgbmc_params = {
    "learning_rate": [0.1, 0.5, 1],
    "n_estimators": [100, 200, 500, 1000],
    "max_depth": [3, 5, 7]
}

grid = GridSearchCV(lgbmc, lgbmc_params, cv=10, verbose=2, n_jobs=-1).fit(X_train, y_train)
grid.best_params_ # Out[13]: {'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 100}

lgbmc_tuned = LGBMClassifier(learning_rate=0.1, max_depth=3, n_estimators=100).fit(X_train, y_train)
y_pred = lgbmc_tuned.predict(X_test)
accuracy_tuned = accuracy_score(y_test, y_pred)

print(f"Accuracy Score: {accuracy}")
print(f"Accuracy Score Tuned: {accuracy_tuned}")