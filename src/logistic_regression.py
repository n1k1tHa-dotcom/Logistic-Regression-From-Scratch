import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def compute_cost(theta, X, y):
    m = len(y)
    y_pred = sigmoid(X @ theta)

    cost = -(1/m) * np.sum(
        y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)
    )

    gradient = (1/m) * (X.T @ (y_pred - y))
    return cost, gradient

def gradient_descent(X, y, theta, alpha, iterations):
    costs = []
    for _ in range(iterations):
        cost, grad = compute_cost(theta, X, y)
        theta = theta - alpha * grad
        costs.append(cost)
    return theta, costs
