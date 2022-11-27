import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error


def generar_modelo_poly(x, y, color='rojo', grado=1):
    # Ajuste del polinomio de grado 'degree' a los datos de entrenamiento x,y
    coeffs = np.polyfit(x, y, deg=grado)
    # Determinar y escribir la forma del polinomio
    p = np.poly1d(np.polyfit(x, y, deg=grado), variable='X')
    print("Polinomio de grado ", grado, " : ")
    print(p)
    print("")

    y_pred = np.polyval(np.poly1d(coeffs), x)
    print('Error cuadrático medio: %.2f\n' % mean_squared_error(y, y_pred))

    y_plot = np.polyval(np.poly1d(coeffs), x)

    plt.plot(x, y_plot, color=color,
             linewidth=2, label="grado% d" % grado)


n_data = 400

# generate random data for poly regression
x = np.random.rand(n_data, 1)
y = 2 + 3 * x + np.random.rand(n_data, 1)

# divide date into training and test sets
n_train = int(n_data * 0.7)

x_train = np.array(x[:n_train]).reshape(-1)
y_train = np.array(y[:n_train]).reshape(-1)

x_test = np.array(x[n_train:]).reshape(-1)
y_test = np.array(y[n_train:]).reshape(-1)

print(x_train)
generar_modelo_poly(x_train, y_train, color='blue', grado=1)
generar_modelo_poly(x_train, y_train, color='grey', grado=4)

# scatter training data
plt.scatter(x_train, y_train, color='black')

# scatter test data
plt.scatter(x_test, y_test, color='red')

# show graph
plt.show()
