import numpy as np
import pandas as pd
from logistic_regression import gradient_descent

data = pd.read_csv("../data/DMV_Written_Tests.csv")

X = data[['DMV_Test_1', 'DMV_Test_2']].values
y = data['Results'].values.reshape(-1, 1)

mean = X.mean(axis=0)
std = X.std(axis=0)
X = (X - mean) / std

X = np.c_[np.ones(X.shape[0]), X]
theta = np.zeros((X.shape[1], 1))

theta, costs = gradient_descent(X, y, theta, 1, 200)

np.save("theta.npy", theta)
np.save("mean.npy", mean)
np.save("std.npy", std)

print("Training completed!")
