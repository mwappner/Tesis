# -*- coding: utf-8 -*-
"""
Clasificador binario (perros y gatos)

"""

#%% Gatos y perros: los archivos

import os, shutil

#directorio original con todo
original_dataset_dir = '/home/marcos/Documents/Probando codigos/Imagenes gatos y perros/train'
#nuevo directorio con cachos chicos de datos
base_dir = '/home/marcos/Documents/Probando codigos/Imagenes gatos y perros/cats_and_dogs_small'

#%% Los directorios (correr siempre)

#directorios para los distintos sets de imagenes
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

#subdirectorios para separar entrenamiento, test y validación
train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')

validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

test_cats_dir = os.path.join(test_dir, 'cats')
test_dogs_dir = os.path.join(test_dir, 'dogs')

#%%Creo directorios y copio imagenes (corer sólo la primera vez: guardo la sección por completitud)

#si ya existen, va a saltar error
os.mkdir(base_dir) #directorio con las imagenes (selección pequeña)
os.mkdir(train_dir) 
os.mkdir(validation_dir)
os.mkdir(test_dir)
os.mkdir(train_cats_dir) #Directory with train cat pictures
os.mkdir(train_dogs_dir) #Directory with train dog pictures
os.mkdir(validation_cats_dir) #Directory with validation cat pictures
os.mkdir(validation_dogs_dir) #Directory with calidation dog pictures
os.mkdir(test_cats_dir) #Directory with test cat pictures
os.mkdir(test_dogs_dir) #Directory with test dog pictures


#puestas a mano la cantidad 

#primeras 1000 para entrenar (gatos)
fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname) #os.path.join() concatena strings de forma inteligente para que sirva para directorios
    dst = os.path.join(train_cats_dir, fname)
    shutil.copyfile(src, dst)

#siguientes 500 para validar (gatos)
fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_cats_dir, fname)
    shutil.copyfile(src, dst) 

#siguientes 500 para test (gatos)
fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_cats_dir, fname)
    shutil.copyfile(src, dst) 

#perros entrenamiento
fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_dogs_dir, fname)
    shutil.copyfile(src, dst)

#perros validación
fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_dogs_dir, fname)
    shutil.copyfile(src, dst)

#perros test
fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_dogs_dir, fname)
    shutil.copyfile(src, dst)

#%% El modelo

from keras import layers
from keras import models

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid')) #binario

#compilo modelo con un calsificador binario
from keras import optimizers
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),metrics=['acc'])
              
#%% Preproceso datos (con los generadores)
              
from keras.preprocessing.image import ImageDataGenerator

#creo generadores reescalando a [0,1]
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

#el generador carga reescalando a 150x150 en camadas de a 20 iamgenes
train_generator = train_datagen.flow_from_directory(
            train_dir,target_size=(150, 150),
            batch_size=20,
            class_mode='binary') #como voy a hacer clasificación binaria, necesito los labels binarios
validation_generator = val_datagen.flow_from_directory(
            validation_dir,target_size=(150, 150),
            batch_size=20,
            class_mode='binary')
            
#%% Entreno
            
history = model.fit_generator(train_generator, #en vez de darle un tensor con los datos, le doy el generador
                              steps_per_epoch=100, #como tengo 2000 imagenes y camadas de 20 imagenes cada, son 100 camadas
                              epochs=30,
                              validation_data=validation_generator,
                              validation_steps=50) #1000 de validación (500 de cada categoria)
                              
#%% Guardo modelo
                              
model.save('/home/marcos/Documents/Probando codigos/cats_and_dogs_small_1.h5')

#%% Muestro curvas (flojo overfitting)

import matplotlib.pyplot as plt

#voy a graficar precisión y pérdida en función de la época, para el conjunto 
#de entrenamiento y para el de validación (para monitorear overfiteo y eso)
acc = history.history['acc'] 
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1) #vector de epocas

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

#%% Evaluo modelo (a pesar del overfitting)

#debería darme algo parecido a la validación

#creo el mismo generador con las mismas características que el de validacion
test_generator = val_datagen.flow_from_directory(test_dir,
                                                  target_size=(150, 150),
                                                  batch_size=20,
                                                  class_mode='binary')
test_loss, test_acc = model.evaluate_generator(test_generator, steps=50)
print('test acc:', test_acc)