# Librerías
import graphviz
import numpy as np
import pandas as pd
from sklearn import tree

# Cargar base de datos de artistas de música
artists_billboard = pd.read_csv("Base_Artistas_Billboard.csv")

# Seleccionamos la parte de los datos codificados (encoded)
datos = artists_billboard.iloc[:, 10:]

# Definir las variables de entrenamiento
y_train = datos['top']
X_train = datos.drop(
    ['top', 'anioNacimiento', 'edad_en_billboard'], axis=1).values

# Crear Arbol de decision con profundidad 'depth=4'
depth = 4
decision_tree = tree.DecisionTreeClassifier(criterion='entropy',
                                            min_samples_split=20,
                                            min_samples_leaf=5,
                                            max_depth=depth,
                                            class_weight={1: 3.5})


# Entrenar el modelo
decision_tree.fit(X_train, y_train)

# Generar el gráfico
dot_data = tree.export_graphviz(decision_tree, out_file=None, feature_names=list(datos.drop(
    ['top', 'anioNacimiento', 'edad_en_billboard'], axis=1)), class_names=['No', 'N1 Billboard'], filled=True, rounded=True, special_characters=True)
# Exportar el gráfico a formato PDF
graph = graphviz.Source(dot_data)
graph.render("Arbol_N1_Billboard")

# EVALUACIÓN. Precisión alcanzada por el árbol
acc_decision_tree = np.round(decision_tree.score(X_train, y_train) * 100, 2)
print("Precisión del Árbol de Decisión: ", acc_decision_tree)
print("")


# Probar el árbol con 2 artistas que entraron al billboard 100 en 2017:
# Camila Cabello que llegó al numero 1 con la Canción Havana
# e Imagine Dragons con su canción Believer que alcanzó un puesto 42 pero no llegó a la cima

# Camila Cabello con su canción Havana llego a numero 1 Billboard US en 2017
x_test = pd.DataFrame(columns=('top', 'Est_animo_cod', 'tiempo_cod',
                               'genero_cod', 'tipo_artista_cod', 'edad_cod', 'duracion_cod'))
x_test.loc[0] = (1, 5, 2, 4, 1, 0, 3)
y_pred = decision_tree.predict(x_test.drop(['top'], axis=1))
print("Predicción (Camila Cabello): " + str(y_pred))
y_proba = decision_tree.predict_proba(x_test.drop(['top'], axis=1))
print("Probabilidad de Acierto: " + str(np.round(y_proba[0][y_pred] * 100,
                                                 2))+"%")

# predecir artista Imagine Dragons
# con su canción Believer llego al puesto 42 Billboard US en 2017
x_test = pd.DataFrame(columns=('top', 'Est_animo_cod', 'tiempo_cod',
                               'genero_cod', 'tipo_artista_cod', 'edad_cod', 'duracion_cod'))
x_test.loc[0] = (0, 4, 2, 1, 3, 2, 3)
y_pred = decision_tree.predict(x_test.drop(['top'], axis=1))
print("Predicción (Imaging Dragons): " + str(y_pred))
y_proba = decision_tree.predict_proba(x_test.drop(['top'], axis=1))
print("Probabilidad de Acierto: " +
      str(np.round(y_proba[0][y_pred] * 100, 2))+"%")
