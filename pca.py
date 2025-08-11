import numpy as np

X = np.array([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0],
    [2.0, 3.0, 4.0],
    [6.0, 7.0, 8.0],
    [1.5, 2.5, 3.5]
])
X_centered = X - X.mean(axis=0) # PCA expects data centered around the mean
U, s, Vt = np.linalg.svd(X_centered)
W2d = Vt[:2].T # Matrix containing the first 2 principle components
X_proj = X_centered @ W2d
print("The projected 2d matrix is: \n",X_proj)