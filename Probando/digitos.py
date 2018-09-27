# -*- coding: utf-8 -*-
"""
Práctica con la base de datos MNIST de clasificación multicategoría

"""

from keras import layers
from keras import models


#%% Defino modelo

#capas convolucionales
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

#capas densas
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax')) #multicategoría

#%% Importo imagenes 

#importo digitos y trato los datos
from keras.datasets import mnist
from keras.utils import to_categorical

#importo imagenes de entrenamiento y de testeo
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#%% Miro algunas imagenes

from scipy.misc import imshow
import numpy as np

inicial = 0; #imagen inicial
cant = 30; #cant*cant imagenes en total

vertical = np.ones([1,28]).transpose() * 255 #columna de ceros
horizontal = np.ones([29*cant+1]) *255
todo = np.copy(horizontal)
for k in range(cant):
    fila = np.copy(vertical) #inicio la fila
    
    for l in range(cant):
        fila = np.hstack((fila,train_images[k*cant+l,:,:],vertical))
        
    todo = np.vstack((todo,fila,horizontal))

#para que sea un poco más griss (imshow reescala así que no anda)
todo /= 2
todo = todo.astype(int)

imshow(todo)
#%% Reordeno aimagenes y entreno la red

train_images = train_images.reshape((60000, 28, 28, 1)) #le agrego una columna
train_images = train_images.astype('float32') / 255 #lo hago un double en [0,1]
test_images = test_images.reshape((10000, 28, 28, 1)) #le agrego una columna
test_images = test_images.astype('float32') / 255 #lo hago un double en [0,1]
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
model.compile(optimizer='rmsprop',
loss='categorical_crossentropy',
metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5, batch_size=64)

#%% Testeo

test_loss, test_acc = model.evaluate(test_images, test_labels)
test_acc

#%% Guardo modelo

model.save('MNIST_entero.h5')