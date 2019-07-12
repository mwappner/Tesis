# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 18:02:00 2019

@author: Marcos
"""

import os

from keras import layers, models, optimizers, regularizers
from keras.preprocessing.image import ImageDataGenerator

#%% Parámetros generales

im_size = (300, 200)
BATCH_SIZE = 32
BASE_DIR = 'sintetizados','dnn', 'pad'
train_dir = os.path.join(*BASE_DIR, 'train')
val_dir = os.path.join(*BASE_DIR, 'validate')

# #%% Opción 1
# model = models.Sequential()
# model.add(layers.Conv2D(16, kernel_size=7, strides=2, input_shape=(500, 300, 1),
#           kernel_regularizer=regularizers.l2(0.001)))
# model.add(layers.MaxPooling2D(2))
# model.add(layers.Conv2D(16, kernel_size=3, strides=1, 
#           kernel_regularizer=regularizers.l2(0.001)))
# model.add(layers.MaxPooling2D(2))
# model.add(layers.Conv2D(32, kernel_size=3, strides=1, 
#           kernel_regularizer=regularizers.l2(0.001)))
# model.add(layers.MaxPooling2D(2))
# model.add(layers.Conv2D(64, kernel_size=3, strides=1, 
#           kernel_regularizer=regularizers.l2(0.001)))
# model.add(layers.MaxPooling2D(2))
# model.add(layers.Flatten())
# #model.add(layers.Dropout(0.5)) #baja el overfitting
# model.add(layers.Dense(512, activation='relu'))
# model.add(layers.Dense(2, activation='softmax'))

# model.summary()

#%% Opcion 2

model = models.Sequential()
model.add(layers.Conv2D(16, kernel_size=7, strides=1, input_shape=(*im_size, 1),
          kernel_regularizer=regularizers.l2(0.001)))
model.add(layers.MaxPooling2D(4)) # Será mucho?
model.add(layers.Conv2D(16, kernel_size=5, strides=1, 
          kernel_regularizer=regularizers.l2(0.001)))
model.add(layers.MaxPooling2D(2))
model.add(layers.Conv2D(32, kernel_size=3, strides=1, 
          kernel_regularizer=regularizers.l2(0.001)))
model.add(layers.MaxPooling2D(2))
model.add(layers.Conv2D(64, kernel_size=3, strides=1, 
          kernel_regularizer=regularizers.l2(0.001)))
model.add(layers.MaxPooling2D(2))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5)) #baja el overfitting
model.add(layers.Dense(512, activation='relu'))
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
            epochs=15,
            validation_data=val_generator,
            validation_steps=100) 