# -*- coding: utf-8 -*-
"""
Práctica con la base de datos MNIST de clasificación multicategoría

"""

from keras import layers
from keras import models
from keras.utils import to_categorical
import numpy as np



#%% Importo imagenes 

#importo imagenes de entrenamiento y de testeo
#(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
file = np.load('circulos.npz')
imagenes, cantidades = file['imagen'], file['cant_puntos']

#%% Reordeno aimagenes y entreno la red

imagenes = imagenes.reshape((*imagenes.shape, 1)) #le agrego una columna
imagenes= imagenes.astype('float32') #lo hago un double en [0,1]
cantidades = to_categorical(cantidades)

train_images, train_labels = imagenes[:9000], cantidades[:9000]
test_images, test_labels = imagenes[9000:], cantidades[9000:]

#%% Defino modelo

#capas convolucionales
model = models.Sequential()
model.add(layers.Conv2D(8, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(16, (3, 3), activation='relu'))
#model.add(layers.MaxPooling2D((2, 2)))
#model.add(layers.Conv2D(16, (3, 3), activation='relu'))

#capas densas
model.add(layers.Flatten())
model.add(layers.Dense(32, activation='relu'))
#model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(cantidades.shape[-1], activation='softmax')) #multicategoría


model.compile(optimizer='rmsprop',
loss='categorical_crossentropy',
metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5, batch_size=64)

#%% Testeo

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('test accuracy: ', test_acc)

#%% Guardo modelo

model.save('circulos.h5')