# ==============================================
# ARTIFICIAL NEURAL NETWORK (MLPREGRESSOR) – EXPLANATION
# ----------------------------------------------
# MLPRegressor (Multi-Layer Perceptron) is a feed-forward artificial neural network
# used for regression tasks. It consists of:
#   - Input layer
#   - One or more hidden layers (with neurons)
#   - Output layer
#
# How it works:
#   - Each neuron computes:  y = activation(Wx + b)
#   - The network is trained using backpropagation to minimize prediction error.
#
# Important Parameters:
#   - hidden_layer_sizes: Number of neurons in each hidden layer (e.g. (100,50))
#   - activation: Activation function ('relu', 'tanh', 'logistic', etc.)
#   - solver: Optimization algorithm ('adam', 'sgd')
#   - alpha: L2 regularization (prevents overfitting)
#
# Why scale features?
#   Neural networks are sensitive to different feature scales.
#   StandardScaler is used to normalize data (mean=0, std=1)
# ==============================================


# ================================
# Import Required Libraries
# ================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from warnings import filterwarnings
filterwarnings("ignore")

# ================================
# Load and Prepare Dataset
# ================================
df = pd.read_csv("..\Data\Hitters.csv")
df.info()

# Fill missing values in Salary
df["Salary"] = df["Salary"].fillna(df["Salary"].mean())

# Convert categorical variables using one-hot encoding
dms = pd.get_dummies(df[["League", "Division", "NewLeague"]])
X_ = df.drop(columns=["Salary", "League", "Division", "NewLeague"])
X = pd.concat([X_, dms[['League_N', 'Division_W', 'NewLeague_N']]], axis=1)
y = df["Salary"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# ================================
# Feature Scaling (Important for Neural Networks)
# ================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ================================
# Initial MLPRegressor Model (Default Parameters)
# ================================
mlp_model = MLPRegressor(random_state=42).fit(X_train_scaled, y_train)

y_pred = mlp_model.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Initial MLP RMSE:", rmse)

# ================================
# Hyperparameter Tuning using GridSearchCV
# ================================
mlp_params = {
    "hidden_layer_sizes": [(10, 20), (5, 5), (100, 100)],
    "activation": ['identity', 'logistic', 'tanh', 'relu'],
    "solver": ['sgd', 'adam'],
    "alpha": [0.0001, 0.001, 0.01, 0.1]  # L2 Regularization
}

grid = GridSearchCV(mlp_model, mlp_params, cv=5, n_jobs=-1, verbose=2)
grid.fit(X_train_scaled, y_train)

# ================================
# Best Hyperparameters and Final Model Evaluation
# ================================
print("Best Parameters:", grid.best_params_)
print("Best CV Score:", grid.best_score_)

# Predict on test set using tuned model
y_pred = grid.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Tuned MLP RMSE:", rmse)
print("Tuned MLP R2 Score:", r2)
