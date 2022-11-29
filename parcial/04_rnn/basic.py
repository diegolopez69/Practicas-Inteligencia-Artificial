import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def perceptron(W, X):
    s = np.sum(W*X)
    print("La suma w1.x1+w2.x2+w3.x3 es: ", s)
    return sigmoid(s)

x1 = 0.9
x2 = 0.3
x3 = 0.5
X = [x1, x2, x3]
X = np.column_stack((x1, x2, x3))
print("Inputs :", X)

w1 = 1
w2 = 0.4
w3 = 0.3
W = [w1, w2, w3]
print("Pesos :", W)

y = perceptron(W, X)
print("El output 'y' de la Red Neuronal es: ", y)
