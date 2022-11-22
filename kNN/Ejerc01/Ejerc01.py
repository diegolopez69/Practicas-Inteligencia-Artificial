# Importar librerías
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Lectura de la base de datos "Base_diabetes.csv"
df = pd.read_csv("Base_diabetes.csv")

# Visualizar las primera 5 filas de la tabla de datos (DataFrame)
print(df.head())

# Ver el tamaño de la tabla de datos
print(df.shape)

# La base de datos tiene 768 registros, con 8 atributos y 1 etiqueta
# Atributos: Embarazos, Glucosa, Presión_sangre, Espesor_piel, Insulina, BMI, Histórico,Edad
# Etiqueta: Categoría (1-> Tiene Diabetes     0-> No tiene diabetes)

# Crear los arrays para X e y
X = df.drop('Categoria', axis=1).values
y = df['Categoria'].values

# Dividir el conjunto de datos en entrenamiento y test
# Utilizar el método 'train_test_split' de la librería sklearn

# Considerar como test un 40 % (test_size=0.4) del total del conjunto de datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y)

# Crear el clasificador basado en KNN (k-Nearest Neighbors)

# Crear los arrays donde registramos las precisiones para entrenamiento y test
neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

# Realizar el cálculo con diferentes valores de 'k'
# para identificar el valor de k que da mejores resultados
for i, k in enumerate(neighbors):
    # Configurar el clasificador KNN con 'k' vecinos (neighbors)
    knn = KNeighborsClassifier(n_neighbors=k)

    # Ajustar el modelo a los datos de entrenamiento
    knn.fit(X_train, y_train)

    # Registrar las precisiones para los datos de entrenamiento
    train_accuracy[i] = knn.score(X_train, y_train)

    # Registrar las precisiones para los datos de test
    test_accuracy[i] = knn.score(X_test, y_test)

# La máxima precisión del ajuste con los valores de test se consigue con k=7
# Configurar el clasificador KNN con '7' vecinos (neighbors)
knn = KNeighborsClassifier(n_neighbors=7)

# Ajustar el modelo a los datos de entrenamiento
knn.fit(X_train, y_train)

# ERRORES/METRICAS
# Calcular la matriz de confusion (confusion matrix)
# Para ver sobre el conjunto de test los resultados

# Importar el método

# Predecir etiquetas sobre el conjunto de test
y_pred = knn.predict(X_test)

# Obtener la matriz de confusión
cm = confusion_matrix(y_test, y_pred)
print(cm)
print()
print("Verdaderos positivos (VP):", cm[1][1])
print("Verdaderos negativos (VN):", cm[0][0])
print("Falsos positivos (FP):", cm[0][1])
print("Falsos negativos (FN):", cm[1][0])

# Informe de clasificación

print()
print(" Informe de clasificación (sobre datos de test")
print(classification_report(y_test, y_pred))

# Curva característica (Curva ROC)
# Curva de falsos positivos (FP) frente a verdaderos positivos (VP)

# Predecir probabilidades sobre el conjunto de test
y_pred_proba = knn.predict_proba(X_test)[:, 1]

# Calcular los ratios de falsos positivos (fpr) y verdadero positivo (tpr)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

# Dibujar la curva ROC
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='Knn')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('Knn(n_neighbors=7) Curva ROC')
plt.show()

# Calcular el área por encima de la curva ROC
roc_auc_score(y_test, y_pred_proba)

# Predecir para un nuevo paciente
# Atributos:Atributos = [Embarazos: 1; Glucosa: 100; Presión en sangre: 70, Espesor de la piel: 30
# Insulina: 90, IMC:  0.3; Histórico: 0.2; Edad: 45]

y_pred_nuevo = knn.predict([[1, 100, 70, 30, 90, 0.3, 0.2, 45]])

if y_pred_nuevo == 1:
    print("El paciente tiene diabetes")
else:
    print("El paciente no tiene diabetes")
