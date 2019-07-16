# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 18:02:00 2019

@author: Marcos
"""

import os

#esto es para poder correr remotamente
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

from keras import layers, models, optimizers, regularizers
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

from utils import new_name, contenidos

#%% Parámetros generales

im_size = (300, 200)
BATCH_SIZE = 32
BASE_DIR = 'sintetizados','dnn', 'stretch'
train_dir = os.path.join(*BASE_DIR, 'train')
val_dir = os.path.join(*BASE_DIR, 'validate')
test_dir = os.path.join(*BASE_DIR, 'test')
ori_dir = os.path.join(*BASE_DIR, 'originales')

nombre_guardado = 'modelos/byc_peque_stretch'
nombre_guardado = new_name(nombre_guardado)
os.makedirs(nombre_guardado)

#%% Defino el modelo
model = models.Sequential()
model.add(layers.Conv2D(4, kernel_size=5, strides=2, input_shape=(*im_size, 1),
          kernel_regularizer=regularizers.l2(0.001)))
model.add(layers.MaxPooling2D(4)) # Será mucho?
model.add(layers.Conv2D(4, kernel_size=3, strides=1, 
          kernel_regularizer=regularizers.l2(0.001)))
model.add(layers.MaxPooling2D(2))
model.add(layers.Conv2D(8, kernel_size=3, strides=1, 
          kernel_regularizer=regularizers.l2(0.001)))
# model.add(layers.MaxPooling2D(2))
# model.add(layers.Conv2D(16, kernel_size=3, strides=1, 
#           kernel_regularizer=regularizers.l2(0.001)))
model.add(layers.MaxPooling2D(2))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5)) #baja el overfitting
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(2, activation='softmax'))

# model.summary()

#%% Compilo y armo generadores

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Uso un generator para train y uno para test por si decido usar augmentation
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

generator_params = dict(target_size=im_size, 
                        batch_size=BATCH_SIZE,
                        color_mode='grayscale', 
                        class_mode='categorical')

train_generator = train_datagen.flow_from_directory(train_dir, **generator_params)
val_generator = train_datagen.flow_from_directory(val_dir, **generator_params)

history=model.fit_generator(
            train_generator,
            steps_per_epoch=100,
            epochs=8,
            validation_data=val_generator,
            validation_steps=100) 

model.save(nombre_guardado + '/byc.h6')

acc=history.history['acc']
val_acc=history.history['val_acc']
loss=history.history['loss']
val_loss=history.history['val_loss']

# Grafico y guardo accuracy y loss
epochs=range(1,len(acc)+1)
plt.plot(epochs, acc, 'o', label='training acc')
plt.plot(epochs, val_acc, label='validation acc')
plt.title('training and validation accuracy')
plt.legend()
plt.savefig(nombre_guardado + '/Accuracy.jpg')
plt.close()

plt.figure()
plt.plot(epochs, loss, 'o', label='training loss')
plt.plot(epochs, val_loss, label='validation loss')
plt.title('training and validation loss')
plt.legend()
plt.savefig(nombre_guardado + '/Loss.jpg')
plt.close()

# # Test
# test_generator = test_datagen.flow_from_directory(test_dir, **generator_params)
# test_loss, test_acc = model.evaluate_generator(test_gen)
# print('Test accuracy: ', test_acc)

# Pruebo con los originales
categorias = {k:v for v, k in train_generator.class_indices.items()}
imgs_paths = contenidos(ori_dir)
for path in imgs_paths:
	img = image.load_img(path, target_size=im_size, grayscale=True)
	x = image.img_to_array(img)
	x = np.expand_dims(x,axis=0)
	preds = model.predict(x)
	preds = np.squeeze(preds)
	print(os.path.basename(path))
	print('{}: {:.0f}% \t {}: {:.0f}%'.format(categorias[0], preds[0]*100, categorias[1], preds[1]*100))