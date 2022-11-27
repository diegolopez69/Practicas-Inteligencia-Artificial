from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# read file and save it as csv
read_file = pd.read_csv('./Base_frutas.txt', sep="\t")
read_file.to_csv('./Base_frutas.csv', index=None,
                 columns=['nombre', 'masa', 'anchura', 'altura'])

# read csv file
df = pd.read_csv('./Base_frutas.csv')

# divide dataframe into X (attributes) and y (label)
X = df.drop('nombre', axis=1).values
y = df['nombre']

# divide dataframe into training (75%) and test (25%) data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=37, stratify=y)

# instantiate the kNN classifier with n neighbors
knn = KNeighborsClassifier(n_neighbors=5)

# fit the model with the training data
knn.fit(X_train, y_train)

# predict for the test data
y_pred = knn.predict(X_test)
print("Predicciones del Test:\n {}".format(y_pred))

# calculate accuracy
print("Test score: {:.2f}".format(np.mean(y_pred == y_test)))
print("Test score: {:.2f}".format(knn.score(X_test, y_test)))

# predict for a new data
prediction = knn.predict([[80, 6, 5]])

if prediction == "manzana":
    print("La especie es manzana")
elif prediction == "pera":
    print("La especie es pera")
elif prediction == "naranja":
    print("La especie es naranja")
else:
    print("La especie es limon")

# plot pairplot
sns.pairplot(df, hue="nombre", markers=["o", "s", "D", "p"])
fig = plt.figure(figsize=(10, 7))

# plot heatmap
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.2)
plt.show()
