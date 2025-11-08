# =====================================================
# LOGISTIC REGRESSION – CLASSIFICATION MODEL
# -----------------------------------------------------
# Logistic Regression is a supervised learning algorithm
# used for binary classification problems.
# It models the probability that a given input (X)
# belongs to class 1 (positive class).
#
# Mathematical Function:
#     P(y = 1 | X) = 1 / (1 + e^-(b0 + b1*x1 + ... + bn*xn))
#
# Why Logistic Regression?
#   ✓ Simple and interpretable
#   ✓ Outputs probabilities (not just classes)
#   ✓ Works well for linearly separable data
#
# Evaluation Metrics Used in This Script:
#   • Accuracy Score        – Correct predictions / total predictions
#   • Confusion Matrix      – TP, FP, TN, FN values
#   • Classification Report – Precision, Recall, F1-Score
#   • ROC & AUC             – Ability of model to separate classes
#   • Cross-Validation      – Model stability on unseen data
# =====================================================

# ================================
# Import Required Libraries
# ================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve
)
from sklearn.linear_model import LogisticRegression

# ================================
# Load Dataset
# ================================
df = pd.read_csv("../Data/diabetes.csv")

# Features and Target Variable
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# ================================
# Train-Test Split
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ================================
# Model Training
# ================================
lr = LogisticRegression(random_state=42, solver="liblinear")
lr.fit(X_train, y_train)

# Model Parameters
print("Intercept:", lr.intercept_)
print("Coefficients:", lr.coef_)

# ================================
# Predictions & Probability Scores
# ================================
y_pred_proba = lr.predict_proba(X_test)[:, 1]      # Probabilities for class = 1
y_pred = (y_pred_proba > 0.5).astype(int)         # Convert to 0/1 based on threshold

# ================================
# Evaluation Metrics
# ================================
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Accuracy Score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy Score:", accuracy)

# Classification Report
print("Classification Report:\n", classification_report(y_test, y_pred))

# Cross-Validation Score (10-fold)
cv_score = cross_val_score(lr, X_train, y_train, cv=10).mean()
print("Cross-Validation Score (CV=10):", cv_score)

# ================================
# ROC Curve & AUC Score
# ================================
logit_auc = roc_auc_score(y_test, y_pred_proba)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

plt.figure()
plt.plot(fpr, tpr, label="ROC Curve (AUC = %0.2f)" % logit_auc)
plt.plot([0, 1], [0, 1], "r--")       # Diagonal (random guess line)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (Logistic Regression)")
plt.legend(loc="lower right")
plt.show()
