import numpy as np
from logistic_regression import sigmoid

theta = np.load("theta.npy")
mean = np.load("mean.npy")
std = np.load("std.npy")

test = np.array([50, 79])
test = (test - mean) / std
test = np.insert(test, 0, 1)

probability = sigmoid(test @ theta)
print("Probability of passing:", probability[0])
