import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split ,  GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

df = pd.read_csv("..\Data\diabetes.csv")

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

mlp = MLPClassifier().fit(X_train, y_train)
y_pred = mlp.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

mlp_params = {
    "alpha": [0.1, 0.5, 1, 3],
    "hidden_layer_sizes": [(10, 10), (100, 100), (100, 100, 100), (3, 5)]
}

grid = GridSearchCV(mlp, mlp_params, cv=10, verbose=2, n_jobs=-1).fit(X_train, y_train)
grid.best_params_ # {'alpha': 3, 'hidden_layer_sizes': (100, 100)}

mlp_tuned = MLPClassifier(alpha=3, hidden_layer_sizes=(100, 100)).fit(X_train, y_train)
y_pred = mlp_tuned.predict(X_test)
accuracy_tuned = accuracy_score(y_test, y_pred)

print(f"Accuracy Score: {accuracy}")
print(f"Accuracy Score Tuned: {accuracy_tuned}")