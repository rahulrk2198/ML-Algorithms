import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.datasets import load_diabetes

def linear_regression(X , y):
    theta = np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T),y)
    return theta

def main():
    #path = "C:/Users/HP/Documents/Python Scripts/Spreadsheets/homeprices.csv"
    diabetes = load_diabetes()
    data = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    data["target"] = diabetes.target
    y_df = data["target"]
    data = data.drop("target",axis=1)
    X = np.concatenate([np.ones((data.shape[0],1)),data.values], axis=1)
    y = y_df.values
    theta = linear_regression(X,y)
    y_pred = np.dot(X, theta)
    print("Accuracy : ", r2_score(y, y_pred))

main()