# Librerías
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def perceptron(W, X):
    # Multiplicar W*X elemento a elemento los vectores = w1*x1+w2*x2+w3*x3
    s = np.sum(W*X)
    print("La suma w1.x1+w2.x2+w3.x3 es: ", s)
    # Aplicar la función de activación (por ejemplo tanh) al resultado 's'
    return sigmoid(s)

# Inputs 'x'
x1 = 0.9
x2 = 0.3
x3 = 0.5
X = [x1, x2, x3]
X = np.column_stack((x1, x2, x3))   # en forma matricial
print("Inputs :", X)

# Pesos 'w'
w1 = 1
w2 = 0.4
w3 = 0.3
W = [w1, w2, w3]
print("Pesos :", W)

y = perceptron(W, X)
print("El output 'y' de la Red Neuronal es: ", y)
