import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.datasets import load_diabetes
def batch_gradient_descent(X, y, alpha, theta , epochs):
    m = X.shape[0]
    for j in range(epochs):
        diff = np.dot(X , theta) - y
        prod = np.dot(X.T, diff) / m
        theta = theta - alpha * prod
        
    return theta

def main():
    diabetes = load_diabetes()
    data = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    data["target"] = diabetes.target
    y_df = data["target"]
    data = data.drop("target",axis=1)
    X = np.concatenate([np.ones((data.shape[0],1)),data.values], axis=1)
    y = y_df.values
    theta = np.zeros(X.shape[1])
    theta = batch_gradient_descent(X , y, 0.01,theta, 1000)
    y_pred = np.dot(X , theta)
    print("Accuracy: ", r2_score(y, y_pred))
main()