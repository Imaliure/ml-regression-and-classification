import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split , cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix,roc_auc_score,roc_curve
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("..\Data\diabetes.csv")

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_proba = knn.predict_proba(X_test)[:, 1]

cm = confusion_matrix(y_test, y_pred_proba > 0.5)
accuracy = accuracy_score(y_test, y_pred_proba > 0.5)

knn_params = {"n_neighbors": range(1, 20)}

grid = GridSearchCV(knn, knn_params, cv=10, n_jobs=-1,verbose=2)
grid.fit(X_train, y_train)
grid.best_params_ # Out[33]: {'n_neighbors': 12}

knn_tuned = KNeighborsClassifier(n_neighbors=12)
knn_tuned.fit(X_train, y_train)
y_pred_proba = knn_tuned.predict_proba(X_test)[:, 1]
accuracy_tuned = accuracy_score(y_test, y_pred_proba > 0.5)

print(f"Accuracy Score: {accuracy}")
print(f"Accuracy Score Tuned: {accuracy_tuned}")