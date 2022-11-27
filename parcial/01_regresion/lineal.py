import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

n_data = 60

# generate random data for linear regression
x = np.random.rand(n_data, 1)
y = 2 + 3 * x + np.random.rand(n_data, 1)

# divide date into training and test sets
n_train = int(n_data * 0.7)

x_train = np.array(x[:n_train])
y_train = np.array(y[:n_train])

x_test = np.array(x[n_train:])
y_test = np.array(y[n_train:])

# instanciate the model
model = linear_model.LinearRegression()

# train model
model.fit(x_train, y_train)
y_pred = model.predict(x_train)

# print training model parameters
print('Parametros de los valores de train: ')
print('Pendiente: \n', model.coef_)
print('Corte con el eje Y (en X=0): \n', model.intercept_)
print('Error cuadrático medio: %.2f' % mean_squared_error(y_train, y_pred))
print('Coeficiente de Correlacción: %.2f' % model.score(x_train, y_train))
print('Recta de Regresión Lineal (y=t0+t1*X): y = %.2f + %.2f * x\n' % (model.intercept_, model.coef_))

# predict test values
y_pred_test = model.predict(x_test)

# print test model parameters
print('Predicción de los valores de test: ')
print('Error cuadrático medio: %.2f' % mean_squared_error(y_test, y_pred_test))
print('Coeficiente de Correlacción: %.2f' % model.score(x_test, y_test))

# predict a value x=0.5
predict = model.predict([[0.5]])
print('y(0.5) = %.2f' % predict)

# scatter training data
plt.scatter(x_train, y_train, color='black')

# scatter test data
plt.scatter(x_test, y_test, color='red')

# plot regression line
plt.plot(x_train, y_pred, color='blue', linewidth=3)

# show graph
plt.title("X vs y (Regresión Lineal)")
plt.xlabel("X")
plt.ylabel("y")
plt.show()
