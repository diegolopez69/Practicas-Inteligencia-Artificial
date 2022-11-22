# Librerías
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Cargar los datos y generar inputs (X) y outputs (y) de la Red Neuronal
data = np.loadtxt('Datos.txt')
X = data[:, 1:]
y = data[:, 0]

# Mostrar los datos
colors = ['red', 'blue']
plt.scatter(X[:, 0], X[:, 1], c='g')   # todos los datos en verde
plt.scatter(X[y > 0, 0], X[y > 0, 1], c=colors[0])    # en rojo los '1'
plt.scatter(X[y == 0, 0], X[y == 0, 1], c=colors[1])  # en azul los '0'
plt.show()

# Dividir los datos 'X' para entrenamiento y validación(test)
ntrain = int(3*len(y)/4)
X_train = X[:ntrain, :]
y_train = y[:ntrain]
X_test = X[ntrain:, :]
y_test = y[ntrain:]
print("Entrenamiento:   X ", np.shape(X_train), "   y ", np.shape(y_train))
print("Test:   X ", np.shape(X_test), "   y ", np.shape(y_test))

# Crear el clasificador basado en Redes Neuronales
# definir numero de capas y de hidden nodes, función de activación
mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10),
                    activation='relu', max_iter=500)

# Entrenamiento y ajuste de la red neuronal
mlp.fit(X_train, y_train)

# Gráfica para ver rendimiento del entranamiento
plt.plot(mlp.loss_curve_)
# mlp.loss_curve nos da una idea de como converge con los datos de entrenamiento
plt.xlabel('Iteracción')
plt.ylabel('Función de coste')
plt.show()

# Validar (test) la red neuronal
y_pred = mlp.predict(X_test)

# Evaluar la red neuronal
print("Matriz de confusión")
print(confusion_matrix(y_test, y_pred))
# C_ii Positivos verdaderos, C_ij!=0 Falsos Positivos
print()
print("Falsos positivos (0): ", confusion_matrix(y_test, y_pred)[0][1])
print("Falsos positivos (1): ", confusion_matrix(y_test, y_pred)[1][0])
print()

# Aplicar métrica para clasificar los datos usados para validación (test)
print("Clasificación de los resultados de la Validación(test)")
print("precision = num detecciones correctas / numero detecciones")
print("recall=num detecciones correctas / numero total de objetos en esa clase")
print()
print(classification_report(y_test, y_pred))

# Gráfica para comprobar datos de validación que son falsos positivos
plt.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], c='r')
plt.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], c='b')
plt.scatter(X_test[(y_test == 1) & (y_pred == 0), 0],
            X_test[(y_test == 1) & (y_pred == 0), 1], c='orange')

plt.scatter(X_test[(y_test == 0) & (y_pred == 1), 0],
            X_test[(y_test == 0) & (y_pred == 1), 1], c='orange')
plt.show()
