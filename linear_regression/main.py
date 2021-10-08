from sklearn import metrics
from sklearn import datasets 
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np 

boston = datasets.load_boston()

X = boston.data 
y = boston.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Linear algorithm
l_reg = linear_model.LinearRegression()

plt.scatter(X.T[5], y)
plt.show()

# Train
model = l_reg.fit(X_train, y_train)
predictions = model.predict(X_test)

print(f"Predictions: {predictions}")
print(f"R^2 value: {l_reg.score(X, y)}")
print(f"coeff: {l_reg.coef_}")
print(f"intercept: {l_reg.intercept_}")