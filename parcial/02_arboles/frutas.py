# create a decision tree
from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import graphviz
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

frutas = pd.read_csv('Base_frutas.csv')

# take masa, anchura, altura into x
x = frutas.drop(['nombre'], axis=1).to_numpy()
print(x)

# take nombre into y
y = frutas['nombre'].to_numpy()
print(y)

# split data into train and test
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=0)

# create a decision tree classifier
classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)

classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

print(accuracy_score(y_test, y_pred))

# masa, anchura, altura
pred = classifier.predict([[172., 7.4, 7]])
print(pred)

# visualize
dot_data = tree.export_graphviz(classifier, out_file=None, feature_names=frutas.columns[1:], class_names=frutas['nombre'].unique(
), filled=True, rounded=True, special_characters=True)

graph = graphviz.Source(dot_data)
graph.render("frutas")

graph.view()
