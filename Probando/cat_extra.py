# -*- coding: utf-8 -*-
"""
Miro qué onda con la categoría extra

"""

import os
from keras import models
import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical

#importo imagenes de entrenamiento y de testeo
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1)) #le agrego una columna
train_images = train_images.astype('float32') / 255 #lo hago un double en [0,1]
test_images = test_images.reshape((10000, 28, 28, 1)) #le agrego una columna
test_images = test_images.astype('float32') / 255 #lo hago un double en [0,1]
train_labels_cat = to_categorical(train_labels)
test_labels_cat = to_categorical(test_labels)

#hago als label para la categoría extra
cant_test = test_labels.shape[0]
test_labels_extra = np.concatenate((test_labels_cat,np.zeros((cant_test,1))),axis=1)


os.chdir('/home/marcos/Documents/Probando codigos')
modelo = models.load_model('MNIST_entero.h5')
modelo_e = models.load_model('MNIST_con_uno_mas.h5')

#%% Veo que agregar categoría no empeora resultados


test_acc = modelo.evaluate(test_images, test_labels_cat)[1]
test_acc_e = modelo_e.evaluate(test_images, test_labels_extra)[1]

print('Modelo original:',test_acc)
print('Modelo con categoría extra:',test_acc_e)
#%% Predigo para ruido en el que no tiene cat extra
#función que predice, para acortar notación
pred = lambda datos:np.argmax(modelo.predict_on_batch(datos),axis=1)

batch = 20 #cant de imagenes a testear

#tomo batch imagenes seguidas empezando en un punto aleatorio, guardo sus labels
desde = np.random.randint(0,cant_test-batch)
prueba_im = test_images[desde:desde+batch,:,:,:]
prueba_lab = test_labels[desde:desde+batch]
resultado = pred(prueba_im)
print(resultado,'resultado de la predicción')
print(prueba_lab, 'real')

ruido = np.random.rand(batch,28,28,1)
ruido = ruido.astype('float32')
res_ruido = pred(ruido)
print(res_ruido,'ruido: cosas aleatorias, mucho 8 (?)') 

#%% Predigo apra ruido en el que tiene extra

pred_e = lambda datos:np.argmax(modelo_e.predict_on_batch(datos),axis=1)
res_ruido_e = pred_e(ruido)
print(res_ruido,'ruido según el modelo extra')

#%% Pruebo con MUCHO ruido

cant = 10000
MUCHO_ruido = np.random.rand(cant,28,28,1)
MUCHO_ruido = MUCHO_ruido.astype('float32')
labels_ruido = to_categorical(np.ones((cant))*10)

test_acc_r = modelo_e.evaluate(MUCHO_ruido, labels_ruido)[1]
print('Predicción del ruido:',test_acc_r)
print('Veo que aprende que la cat. extra es una categoría prohibida')
