# -*- coding: utf-8 -*-
"""
Práctica con la base de datos MNIST de clasificación multicategoría
Agrego una categoría a la que no le doy datos de entrenamiento, así veo
qué aprende a poner ahí. La cant. de categorías se enseña con el tamaño
de los vectores de labels y se define en el sofmax

"""

from keras import layers
from keras import models


#%% Defino model_eo

#capas convolucionales
model_e = models.Sequential() #_e por extra
model_e.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model_e.add(layers.MaxPooling2D((2, 2)))
model_e.add(layers.Conv2D(64, (3, 3), activation='relu'))
model_e.add(layers.MaxPooling2D((2, 2)))
model_e.add(layers.Conv2D(64, (3, 3), activation='relu'))

#capas densas
model_e.add(layers.Flatten())
model_e.add(layers.Dense(64, activation='relu'))
model_e.add(layers.Dense(11, activation='softmax')) #multicategoría

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
#%% Reordeno imagenes 

train_images = train_images.reshape((60000, 28, 28, 1)) #le agrego una columna porque los filtros la necesitan
train_images = train_images.astype('float32') / 255 #lo hago un double en [0,1]
test_images = test_images.reshape((10000, 28, 28, 1)) #le agrego una columna
test_images = test_images.astype('float32') / 255 #lo hago un double en [0,1]
train_labels = to_categorical(train_labels) #lo paso a categorico
test_labels = to_categorical(test_labels) #lo paso a categorico

cant_train = train_labels.shape[0]
cant_test = test_labels.shape[0]

train_labels_nuevas = np.concatenate((train_labels,np.zeros((cant_train,1))),axis=1)
test_labels_nuevas = np.concatenate((test_labels,np.zeros((cant_test,1))),axis=1)

#%% Esta forma es para el caso de labels binarios

##agrego categoría vacia (categoria 10)
#cat_extra = [[1,0]] # "no pertenece a la categoría 10 (la extra)"
#cant_train = train_labels.shape[0]
#cant_test = test_labels.shape[0]
#
#cat_extra_train = np.repeat(cat_extra,cant_train,axis=0) #repito el vector 
#cat_extra_test = np.repeat(cat_extra,cant_test,axis=0) #repito el vector 
#
#train_labels = np.concatenate((train_labels,np.expand_dims(cat_extra_train,axis=1)),axis=1)
#test_labels = np.concatenate((test_labels,np.expand_dims(cat_extra_test,axis=1)),axis=1)


#%% Entreno la red
model_e.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model_e.fit(train_images, train_labels_nuevas, epochs=5, batch_size=64)

#%% Testeo

test_loss, test_acc = model_e.evaluate(test_images, test_labels_nuevas)
test_acc

#%% Guardo model_eo

model_e.save('MNIST_con_uno_mas.h5')
