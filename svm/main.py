from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import pandas as pd

data = datasets.load_iris()

X = data.data
y = data.target 

# For indexing purposes
classes = ['Iris Setosa', 'Iris Versicolour', 'Iris Virginica']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.2)

# SVM model
model = SVC()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = metrics.accuracy_score(y_test, predictions)

print(f"Predictions: {predictions}")
print(f"Accuracy: {accuracy}")

