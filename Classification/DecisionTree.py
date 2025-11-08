import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split ,  GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("..\Data\diabetes.csv")

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dt_model = DecisionTreeClassifier().fit(X_train, y_train)
y_pred = dt_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

dt_params = {
    "max_depth": [1, 3, 5, 8, 10],
    "min_samples_split": [2, 3, 5, 10, 20],
}

grid = GridSearchCV(dt_model, dt_params, cv=10, verbose=2, n_jobs=-1).fit(X_train, y_train)
grid.best_params_ # Out[4]: {'max_depth': 3, 'min_samples_split': 3}

dt_model_tuned = DecisionTreeClassifier(max_depth=3, min_samples_split=3).fit(X_train, y_train)
y_pred = dt_model_tuned.predict(X_test)
accuracy_tuned = accuracy_score(y_test, y_pred)

print(f"Accuracy Score: {accuracy}")
print(f"Accuracy Score Tuned: {accuracy_tuned}")



