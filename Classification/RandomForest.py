import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split ,  GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


df = pd.read_csv("..\Data\diabetes.csv")

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier().fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

rf_params = {
    "max_depth": [1, 3, 5, 8, 10],
    "n_estimators": [10, 50, 100, 500]
}

grid = GridSearchCV(rf_model, rf_params, cv=10, verbose=2, n_jobs=-1).fit(X_train, y_train)
grid.best_params_ #Out[6]: {'max_depth': 8, 'n_estimators': 500}

rf_model_tuned = RandomForestClassifier(max_depth=8, n_estimators=500).fit(X_train, y_train)
y_pred = rf_model_tuned.predict(X_test)
accuracy_tuned = accuracy_score(y_test, y_pred)

print(f"Accuracy Score: {accuracy}")
print(f"Accuracy Score Tuned: {accuracy_tuned}")