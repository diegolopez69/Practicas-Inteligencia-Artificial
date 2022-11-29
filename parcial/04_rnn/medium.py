import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn import model_selection

# generate 3 random integer sequences X1, X2 and X3 (inputs) and we add them (y, output)
ndatos = 1000
X1 = np.round(np.random.uniform(size=ndatos)*100)
X2 = np.round(np.random.uniform(size=ndatos)*100)
X3 = np.round(np.random.uniform(size=ndatos)*100)

# we pass it to matrix form
X = np.transpose([X1, X2, X3])

# calculate the output (sum of the three numbers)
y = X1+X2+X3

print("Dimensiones: X ", np.shape(X), "   y ", np.shape(y))

# use the sklearn library for training
X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, y, test_size=0.2, random_state=7)

# use the sklearn library method for the neural network
mlp = MLPRegressor(hidden_layer_sizes=(10), max_iter=2000, verbose=True)
# train the neural network
mlp.fit(X_train, y_train)

# predict for the test data
predictions = mlp.predict(X_test)

# evaluate the neural network by calculating the error
print("Correlación: ", np.corrcoef(predictions, y_test))

# visualize the results
plt.plot(predictions, y_test, '.')

# predict for new data
X_pred = [[70, 20, 5]]
y_pred = mlp.predict(X_pred)
print('La suma es', y_pred)

plt.show()
