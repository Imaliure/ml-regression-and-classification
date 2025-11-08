import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split , cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report , roc_auc_score,roc_curve
from sklearn.linear_model import LogisticRegression


df = pd.read_csv("..\Data\diabetes.csv")
df.head()
df["Outcome"].value_counts()
decribe = df.describe().T

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LogisticRegression(random_state=42,solver="liblinear")
lr.fit(X_train, y_train)

lr.intercept_
lr.coef_

y_pred_proba = lr.predict_proba(X_test)[:, 1]

cm = confusion_matrix(y_test, y_pred_proba > 0.5)

accuracy = accuracy_score(y_test, y_pred_proba > 0.5)

cr = classification_report(y_test, y_pred_proba > 0.5)

cv = cross_val_score(lr, X_train, y_train, cv=10).mean()


logit_roc_auc = roc_auc_score(y_test, y_pred_proba)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

plt.figure()
plt.plot(fpr, tpr, label="AUC (area = %0.2f)" % logit_roc_auc)
plt.plot([0, 1], [0, 1], "r--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic")
plt.legend(loc="lower right")
plt.show()

