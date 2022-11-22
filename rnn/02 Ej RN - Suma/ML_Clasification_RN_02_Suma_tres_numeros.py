# Librerías
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn import model_selection

# Generar 3 secuencias de números enteros aleatorios X1, X2 y X3 (inputs) y los sumamos(y, output)
ndatos = 1000
X1 = np.round(np.random.uniform(size=ndatos)*100)
X2 = np.round(np.random.uniform(size=ndatos)*100)
X3 = np.round(np.random.uniform(size=ndatos)*100)
# Lo pasamos a forma matricial
X = np.transpose([X1, X2, X3])

# Calcular el output (suma de los tres números)
y = X1+X2+X3

print("Dimensiones: X ", np.shape(X), "   y ", np.shape(y))

# Utilizar la librería sklearn para entrenamiento
# Seleccionar los que queremos utilizar para entrenamiento y para test
# Dividir para entrenamiento y test (80 % para entrenamiento y 20 % para validación)
X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, y, test_size=0.2, random_state=7)

# Utilizar el método de red neuronal de la librería sklearn
# definimos el número de capas ocultas a utilizar
mlp = MLPRegressor(hidden_layer_sizes=(10), max_iter=500, verbose=True)

# Entrenamiento de la red neuronal
mlp.fit(X_train, y_train)

# Predecir para los datos de test
predictions = mlp.predict(X_test)

# Evaluar la red neuronal calculando el error
print("Correlación: ", np.corrcoef(predictions, y_test))

# Visualizar los resultados
plt.plot(predictions, y_test, '.')

X_pred = [[5, 12, 6]]
y_pred = mlp.predict(X_pred)
print('La suma es', y_pred)
