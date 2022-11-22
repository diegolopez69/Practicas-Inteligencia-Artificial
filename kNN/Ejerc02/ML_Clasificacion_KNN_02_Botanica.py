# Asignatura: Inteligencia Artificial (IYA051)
# Grado en Ingeniería Informática
# Escuela Politécnica Superior
# Universidad Europea del Atlántico

# Caso Práctico (ML_Clasificacion_KNN_Flores)
# Aprendizaje Supervisado. Modelos de Clasificación basado en K-Nearest Neighbors

# El objetivo de esta práctica es construir un modelo de Machine Learning utilizando el algoritmo
# k-Nearest Neighbors para predecir las especies de flores en función de las medidas (largo y ancho) de pétalos y sépalos.

# La información a utilizar para esta práctica es la base de datos 'Iris' que se incluye con la librería 'seaborn'
# y contiene un registro de 150 flores de iris con sus dimensiones y especie de flor

# Librerías
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Cargar la base de datos 'Iris' y dar un primer vistazo con seaborn. Inspeccionar los datos
df = sns.load_dataset("iris")
print(df)
sns.pairplot(df, hue="species", markers=["o", "s", "D"])

# Ver como se correlaccionan
fig = plt.figure(figsize=(10, 7))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.2)
plt.show()

# Dividir el DataFrame 'df' en X (atributos) e y (etiqueta)
# X: contiene las 4 medidas (longitud y anchura de sépalo, y longitud y anchura de pétalo)
# y: contiene el valor de la clase o especie de flor (0-> setosa,  1-> versicolor,  2-> virginica)
X = df.drop('species', axis=1).values
y = df['species']

# Dividir de forma aleatoria el conjunto de datos en entrenamiento (75 %) y test (25 %)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=37, stratify=y)
print("X_train tamaño: {}".format(X_train.shape))
print("y_train tamaño: {}".format(y_train.shape))
print("X_test tamaño: {}".format(X_test.shape))
print("y_test tamaño: {}".format(y_test.shape))

# CLASIFICACIÓN CON KNN
# Crear el clasificador
knn = KNeighborsClassifier(n_neighbors=1)

# Ajustar el modelo con los datos de entrenamiento
knn.fit(X_train, y_train)

# Predecir para los datos de test
y_pred = knn.predict(X_test)
print("Predicciones del Test:\n {}".format(y_pred))

# Evaluar el modelo
print("Test score: {:.2f}".format(np.mean(y_pred == y_test)))
print("Test score: {:.2f}".format(knn.score(X_test, y_test)))

# Predecir para nuevos datos
# Ejemplo:
# Longitud y anchura de sépalo: 7 y 3 cm
# Longitud y anchura de pétalo: 5 y 1 cm
y_pred_nuevo = knn.predict([[7, 3, 1, 1]])

if y_pred_nuevo == "setosa":
    print("La especie es setosa")
elif y_pred_nuevo == "versicolor":
    print("La especie es versicolor")
else:
    print("La especie es virginica")
