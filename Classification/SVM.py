import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split ,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC


df = pd.read_csv("..\Data\diabetes.csv")

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svc = SVC(kernel="linear").fit(X_train, y_train)
y_pred = svc.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

svc_params = {
    "C": [0.1, 0.5, 1, 3]
}

grid = GridSearchCV(svc, svc_params, cv=10, verbose=2, n_jobs=-1).fit(X_train, y_train)
grid.best_params_ # Out[8]: {'C': 1}

svc_tuned = SVC(kernel="linear", C=1).fit(X_train, y_train)
y_pred = svc_tuned.predict(X_test)
accuracy_tuned = accuracy_score(y_test, y_pred)

print(f"Accuracy Score: {accuracy}")
print(f"Accuracy Score Tuned: {accuracy_tuned}")