# Librerías
import graphviz
from sklearn.datasets import load_iris
from sklearn import tree

# Lectura dataset "Iris"
iris = load_iris()

# Crear el clasificador basado en Decision Tree
clf = tree.DecisionTreeClassifier()

# Ajustar el modelo
clf = clf.fit(iris.data, iris.target)

# Importar librería gráfico del Árbol de Decision
# Generar el gráfico
dot_data = tree.export_graphviz(clf, out_file=None,
                                feature_names=iris.feature_names,
                                class_names=iris.target_names,
                                filled=True, rounded=True,
                                special_characters=True)
# Exportar el gráfico a formato PDF
graph = graphviz.Source(dot_data)
graph.render("iris_tree_decision")

# Predecir los resultados
# Longitud y anchura de sépalo: 7 y 3 cm
# Longitud y anchura de pétalo: 5 y 1 cm
#                     LS AS LP AP
pred1 = clf.predict([[2, 3, 3, 1]])
print("Tipo de flor: ", iris.target_names[pred1])
