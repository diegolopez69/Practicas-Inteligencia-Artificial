# Importar librerías a utilizar
from keras.models import model_from_json
from keras import layers
from keras import models
from keras.utils import to_categorical
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

# Cargar la base de datos MNIST y asignar las imágenes y etiquetas de los conjuntos para entrenamiento y prueba
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Las imágenes están codificadas en arrays (0 o 1), y las etiquetas son un array números (0 a 9)
print("X_train shape", X_train.shape)
print("y_train shape", y_train.shape)
print("X_test shape", X_test.shape)
print("y_test shape", y_test.shape)

# Consultar algunas imágenes y etiquetas para entrenamiento
# Visualizar algunos ejemplos
fig = plt.figure()
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.tight_layout()
    plt.imshow(X_train[i], cmap='gray', interpolation='none')
    plt.title("Dígito: {}".format(y_train[i]))
    plt.xticks([])
    plt.yticks([])
fig
plt.show()

# Visualizar un ejemplo de las 60.000 imágenes
digit = X_train[4345]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()

# Antes de realizar el entrenamiento, preparar los datos transformando las imágenes
# iniciales con valores entre 0 y 255 (negro a blanco), a valores binarizados (0 a 1)
X_train = X_train.reshape((60000, 28 * 28))
X_train = X_train.astype('float32') / 255
X_test = X_test.reshape((10000, 28 * 28))
X_test = X_test.astype('float32') / 255

# Preparar también las etiquetas en categorías:
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

# Crear la arquitectura de la red neuronal, formada por dos capas ocultas
# La capa de salida representa las 10 categorías posibles (0 a 9)
network = models.Sequential()

# Capa de entrada y primera capa oculta
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))

# Capas ocultas
#network.add(layers.Dense(512, activation='relu'))

# Capa de salida
network.add(layers.Dense(10, activation='softmax'))

# Definir la función de pérdida, el optimizador y las métricas para monitorizar el entrenamientoy la prueba de validación
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy', metrics=['accuracy'])
# Más información en:
# Optimizer: https://keras.io/optimizers/
# Loss function: https://keras.io/losses/
# Metrics: https://keras.io/metrics/

# Realizar el entrenamiento. Guardar el resultado en una variable denominada ‘history’
history = network.fit(X_train,  y_train_cat,  epochs=5,
                      batch_size=128,  validation_data=(X_test, y_test_cat))

# Visualizar las métricas
fig = plt.figure()
plt.subplot(2, 1, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Precición del modelo')
plt.ylabel('Precision')
plt.xlabel('epoch')
plt.legend(['Entrenamiento', 'Test'], loc='lower right')

plt.subplot(2, 1, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Pérdida del modelo')
plt.ylabel('Pérdida')
plt.xlabel('epoch')
plt.legend(['Entrenamiento', 'Test'], loc='upper right')

plt.tight_layout()

plt.show()

# Comprobar el ajuste o error del modelo respecto del conjunto de prueba
test_loss, test_acc = network.evaluate(X_test, y_test_cat)
print('test_acc:', test_acc)

# Guardar el modelo en formato JSON
model_json = network.to_json()
with open("network.json", "w") as json_file:
    json_file.write(model_json)

# Guardar los pesos (weights) a formato HDF5
network.save_weights("network_weights.h5")
print("Guardado el modelo a disco")

# Leer JSON y crear el modelo
json_file = open("network.json", "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# Cargar los pesos (weights) en un nuevo modelo
loaded_model.load_weights("network_weights.h5")
print("Modelo cargado desde el disco")

# Predecir sobre el conjunto de test
predict_x = loaded_model.predict(X_test)
predicted_classes = np.argmax(predict_x, axis=1)

# Comprobar que predicciones son correctas y cuales no
indices_correctos = np.nonzero(predicted_classes == y_test)[0]
indices_incorrectos = np.nonzero(predicted_classes != y_test)[0]
print()
print(len(indices_correctos), " clasificados correctamente")
print(len(indices_incorrectos), " clasificados incorrectamente")

# Adaptar el tamaño de la figura para visualizar 18 subplots
plt.rcParams['figure.figsize'] = (7, 14)

figure_evaluation = plt.figure()

# Visualizar 9 predicciones correctas
for i, correct in enumerate(indices_correctos[:9]):
    plt.subplot(6, 3, i+1)
    plt.imshow(X_test[correct].reshape(28, 28),
               cmap='gray', interpolation='none')
    plt.title(
        "Pred: {}, Original: {}".format(predicted_classes[correct],
                                        y_test[correct]))
    plt.xticks([])
    plt.yticks([])

# Visualizar 9 predicciones incorrectas
for i, incorrect in enumerate(indices_incorrectos[:9]):
    plt.subplot(6, 3, i+10)
    plt.imshow(X_test[incorrect].reshape(28, 28),
               cmap='gray', interpolation='none')
    plt.title(
        "Pred: {}, Original: {}".format(predicted_classes[incorrect],
                                        y_test[incorrect]))
    plt.xticks([])
    plt.yticks([])

figure_evaluation
