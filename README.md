# This is my progress in machine learning
import pandas as pd
import numpy as np
heart_disease = pd.read_csv("heart-disease.csv")

# Create X (features matrix)
X = heart_disease.drop("target", axis=1)

# Create y (labels)
y = heart_disease["target"]

# Choose the right model and hyperparameters
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()

# We will keep the default hyperparameters
clf.get_params()

# Fit the model to the training data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

y_preds = clf.predict(X_test)

# Evaluate the model on the training data and test data
clf.score(X_train, y_train)

clf.score(X_test, y_test)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(classification_report(y_test, y_preds))
