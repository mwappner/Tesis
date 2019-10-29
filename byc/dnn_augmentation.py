# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 18:02:00 2019

@author: Marcos
"""

import os

#esto es para poder correr remotamente
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

#from keras import layers, models, optimizers, regularizers
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

from utils import new_name, contenidos
from dnn_modelos import switcher

#%% Parámetros generales

im_size = (300, 200) #medidas viejas
#im_size = (200, 300)
BATCH_SIZE = 32

MODO = 'pad'
MODELO = 'mas_profunda' # 'peque','peque_conectada', 'peque_densa', 'media', 'grande', 'grande_shallow', 'profunda', 'mas_profunda', 'asimetrica'

BASE_DIR = 'nuevos', 'dnn chica'
train_dir = os.path.join(*BASE_DIR, 'train')
val_dir = os.path.join(*BASE_DIR, 'validate')
test_dir = os.path.join(*BASE_DIR, 'test')
#ori_dir = os.path.join(*BASE_DIR, 'originales')

nombre_guardado = '_'.join(['modelos/aug2/aug', MODO, MODELO])
nombre_guardado = new_name(nombre_guardado)
os.makedirs(nombre_guardado)

model = switcher[MODELO](im_size)

#%% Compilo y armo generadores

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Uso un generator para train y uno para test por si decido usar augmentation
train_datagen = ImageDataGenerator(rescale=1./255,
                                   width_shift_range=(10, 30), #+/-20px
                                   height_shift_range=50, #+/-25px
                                   brightness_range=(.7, 1.3), #0: totalmente oscura, 1:misma imagen, >1 más brillante
                                   shear_range=.2, #??
                                   zoom_range=(0.8, 1.4), #porcentaje de zoom-in/zoom-out
                                   fill_mode='constant', cval=255, #llenar con ceros cuando haga falta
                                   horizontal_flip=False,
                                   vertical_flip=False,
                                   )

#### Para chequear el augmentation ####
#bents = contenidos(contenidos(train_dir)[0])
#chings = contenidos(contenidos(train_dir)[1])
#
#img = load_img(chings[0], target_size=im_size, color_mode='grayscale')
#x = image.img_to_array(img)
#x = x.reshape((1,) + x.shape)
#
#fig, axarr = plt.subplots(3,4, sharex=True, sharey=True)
# 
#for batch, ax in zip(train_datagen.flow(x, batch_size=1), axarr.flatten()):    
#    ax.imshow(image.array_to_img(batch[0]))
#plt.tight_layout()

#%%
test_datagen = ImageDataGenerator(rescale=1./255)

generator_params = dict(target_size=im_size, 
                        batch_size=BATCH_SIZE,
                        color_mode='grayscale', 
                        class_mode='categorical')

train_generator = train_datagen.flow_from_directory(train_dir, **generator_params)
val_generator = test_datagen.flow_from_directory(val_dir, **generator_params)

history=model.fit_generator(
            train_generator,
            steps_per_epoch=100,
            epochs=20,
            validation_data=val_generator,
            validation_steps=50) 

model.save(nombre_guardado + '/byc_aug.h6')

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
#plt.close()

plt.figure()
plt.plot(epochs, loss, 'o', label='training loss')
plt.plot(epochs, val_loss, label='validation loss')
plt.title('training and validation loss')
plt.legend()
plt.savefig(nombre_guardado + '/Loss.jpg')
#plt.close()

# # Test
test_generator = test_datagen.flow_from_directory(test_dir, **generator_params)
test_loss, test_acc = model.evaluate_generator(test_generator,steps=100)
print('Test accuracy: ', test_acc)

# =========================
# Pruebo con los originales
def cargar_imagen(im_path):
    img = load_img(im_path, target_size=im_size, grayscale=True)
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return x

categorias = {k:v for v, k in train_generator.class_indices.items()}
#imgs_paths = contenidos(ori_dir)
#for path in imgs_paths:
#	x = cargar_imagen(path)
#	preds = model.predict(x)
#	preds = np.squeeze(preds)
#	print(os.path.basename(path))
#	print('{}: {:.0f}% \t {}: {:.0f}%'.format(categorias[0], preds[0]*100, categorias[1], preds[1]*100))

# =========================
# Testeo para cantos reales
ori_dir_chin = os.path.join(test_dir, 'chingolo')
ori_dir_bent = os.path.join(test_dir, 'benteveo')

resultados = {'bent':[], 'ching':[]}
paths = {'bent':ori_dir_bent, 'ching':ori_dir_chin}

with open(nombre_guardado + '/resultados.txt', 'w') as resultados_out:
    for pajaro in paths:
        print(pajaro.upper(), file=resultados_out)
        print('', file=resultados_out)
        for i, path in enumerate(contenidos(paths[pajaro])):
            
            x = cargar_imagen(path)
            
            preds = model.predict(x)
            preds = np.squeeze(preds)
            
            resultados[pajaro].append(preds)
            print(i, '{}: {:.0f}% \t {}: {:.0f}%'.format(
                    categorias[0], preds[0]*100, categorias[1], preds[1]*100),
                    file=resultados_out)
        print('', file=resultados_out)

# Confusion matrix
    cm = np.array(
            [[sum((b>c for b, c in resultados['bent'])), sum((b<c for b, c in resultados['bent']))],
           [sum((b>c for b, c in resultados['ching'])), sum((b<c for b, c in resultados['ching']))]]
            )
    acc = (cm[0,0] + cm[1,1])/(len(resultados['bent']) + len(resultados['ching']))
    err = 1-acc
    recB = cm[0,0]/sum(cm[0])
    recC = cm[1,1]/sum(cm[1])
    precB = cm[0,0]/(cm[0,0]+cm[1,0])
    precC = cm[1,1]/(cm[1,1]+cm[0,1])
    
    # cm contiene:
    #___________________
    # bent TP | ching FP
    # bent FP | ching TP
    
    print('', file=resultados_out)
    
    print(cm, file=resultados_out)
    print(cm)
    print('acc =', np.round(acc, 2), ', err =', np.round(err, 2), file=resultados_out)
    print('recB =', np.round(recB, 2), ', recC =', np.round(recC, 2), file=resultados_out)
    print('precB =', np.round(precB, 2), ', precC =', np.round(precC, 2), file=resultados_out)

np.save(nombre_guardado + '/conf_mat.npy', cm)