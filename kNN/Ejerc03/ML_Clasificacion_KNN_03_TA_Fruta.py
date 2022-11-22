# El código en Python debe presentar como resultados:
# - Un gráfico de la anchura de las frutas (eje X) con la altura (eje Y)
# - El valor del parámetro ‘k’ considerado en el modelo KNN
# - El porcentaje de datos de entrenamiento considerado
# - El error del modelo (utilizar el método ‘score’ sobre los datos de test)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

read_file = pd.read_csv('./Base_frutas.txt', sep="\t")
read_file.to_csv('./Base_frutas.csv', index=None,
                 columns=['nombre', 'masa', 'anchura', 'altura'])

df = pd.read_csv('./Base_frutas.csv')

sns.pairplot(df, hue="nombre", markers=["o", "s", "D", "p"])

# Ver como se correlaccionan
fig = plt.figure(figsize=(10, 7))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.2)
plt.show()

# Dividir el DataFrame 'df' en X (atributos) e y (etiqueta)
# X: contiene las 4 medidas (longitud y anchura de sépalo, y longitud y anchura de pétalo)
# y: contiene el valor de la clase o especie de flor (0-> setosa,  1-> versicolor,  2-> virginica)
X = df.drop('nombre', axis=1).values
y = df['nombre']

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
y_pred_nuevo = knn.predict([[80, 6, 5]])

if y_pred_nuevo == "manzana":
    print("La especie es manzana")
elif y_pred_nuevo == "pera":
    print("La especie es pera")
elif y_pred_nuevo == "naranja":
    print("La especie es naranja")
else:
    print("La especie es limon")
