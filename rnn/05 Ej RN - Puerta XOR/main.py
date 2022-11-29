# Librerías
from keras.models import Sequential
from keras.layers.core import Dense
import numpy as np
print('numpy: %s' % np.__version__)
# de la librería Keras importar el tipo de modelo Sequential y el tipo de capa Dense(la más normal)

# Crear los arrays de entrada y salida
# compuertas XOR. Cuatro entradas [0,0], [0,1], [1,0],[1,1] y sus salidas: 0, 1, 1, 0.
training_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], "float32")
# y las salidas, en el mismo orden
target_data = np.array([[0], [1], [1], [0]], "float32")

# Crear la arquitectura de la red neuronal
# Utilizar un modelo de tipo 'Sequential' para crear capas secuenciales, “una delante de otra”
# Agregamos las capas Dense: entrada con 2 neuronas (XOR), capa oculta (16 neuronas)
# Función de activación utilizar “relu” que da buenos resultados
model = Sequential()
model.add(Dense(16, input_dim=2, activation='relu'))
# Agregamos la capa de salida con función de activación 'sigmoid'
model.add(Dense(1, activation='sigmoid'))

# Definir el tipo de pérdida (loss) a utilizar, el “optimizador” de los pesos de las conexiones
# de las neuronas y las métricas a obtener
model.compile(loss='mean_squared_error', optimizer='adam',
              metrics=['binary_accuracy'])

# Entrenar la red neuronal
# Con 1000 iteraciones de aprendizaje (epochs) de entrenamiento
model.fit(training_data, target_data, epochs=1000)

# Evaluamos el modelo
scores = model.evaluate(training_data, target_data)

# Métricas
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
print(model.predict(training_data).round())
