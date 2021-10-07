import numpy as np 
import pandas as pd 
from sklearn import neighbors, metrics
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('knn_model/car.data')
print(data.head())

X = data[[
    'buying',
    'maint',
    'safety'
]].values

y = data[['class']]
print(X, y)

# X
# Converting data
Le = LabelEncoder()
for i in range(len(X[0])):
    X[:, i] = Le.fit_transform(X[:, i])

# y
label_mapping = {
    'unacc': 0,
    'acc': 1,
    'good': 2,
    'vgood': 3,
}
y['class'] = y['class'].map(label_mapping)


# Creating model
# n_neighbors stands for k-constant
knn = neighbors.KNeighborsClassifier(n_neighbors=25, weights='uniform')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Training model
knn.fit(X_train, y_train)


# Model results
prediction = knn.predict(X_test)
accuracy = metrics.accuracy_score(y_test, prediction)

print(f"Prediction: {prediction}")
print(f"Accuracy: {accuracy}")