import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
def batch_gradient_ascent(X, y, alpha, theta , epochs):
    m = X.shape[0]
    for j in range(epochs):
        z =  np.dot(X , theta)
        g = 1 / (1 + np.exp(-z))  # Can cause overflow when z is large. Will fix in future
        diff = y - g
        prod = np.dot(X.T, diff) / m
        theta = theta + alpha * prod
        
    return theta

def logistic_regression(X , theta):
    z = np.dot(X , theta)
    sigmoid = 1 / (1 + np.exp(-z))
    return sigmoid
def main():
    data = load_breast_cancer()
    X, y = data.data, data.target
    X = np.concatenate([np.ones((X.shape[0],1)),X], axis=1)
    theta = np.zeros(X.shape[1])
    theta = batch_gradient_ascent(X , y, 0.01,theta, 1000)
    y_pred = logistic_regression(X , theta)
    y_pred_output = np.where(y_pred >= 0.5, 1 , 0)
    print("Accuracy: ", accuracy_score(y, y_pred_output))
main()