# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 03:03:18 2019

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
from keras.applications import VGG16

from utils import new_name, contenidos

#%% Parámetros generales

im_size = (300, 200) #medidas viejas
#im_size = (200, 300)
BATCH_SIZE = 32

MODO = 'pad'
MODELO = 'VGG16' # 

BASE_DIR = 'sintetizados','dnn', MODO
train_dir = os.path.join(*BASE_DIR, 'train')
val_dir = os.path.join(*BASE_DIR, 'validate')
test_dir = os.path.join(*BASE_DIR, 'test')
ori_dir = os.path.join(*BASE_DIR, 'originales')

nombre_guardado = '_'.join(['modelos/byc', MODO, MODELO])
nombre_guardado = new_name(nombre_guardado)
os.makedirs(nombre_guardado)

#%%
generator_params = dict(target_size=im_size, 
                        batch_size=BATCH_SIZE,
#                        color_mode='grayscale', 
                        class_mode='categorical')

datagen = ImageDataGenerator(rescale=1./255)


conv_base = VGG16(weights='imagenet', 
                  include_top=False,
                  input_shape=(*im_size, 3))

def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, 9, 6, 512))
    labels = np.zeros(shape=(sample_count, 2)) # 2 porque uso categorical
    generator = datagen.flow_from_directory(directory, **generator_params)
    
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * BATCH_SIZE : (i + 1) * BATCH_SIZE] = features_batch
        labels[i * BATCH_SIZE : (i + 1) * BATCH_SIZE] = labels_batch
        i += 1
        if i * BATCH_SIZE >= sample_count:
            break
    return features, labels

train_features, train_labels = extract_features(train_dir, 4000)
validation_features, validation_labels = extract_features(val_dir, 1000)

train_features = np.reshape(train_features, (4000, 9 * 6 * 512))
validation_features = np.reshape(validation_features, (1000, 9 * 6 * 512))


model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_dim=9 * 6 * 512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(2, activation='softmax'))

model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
loss='binary_crossentropy',
metrics=['acc'])
history = model.fit(train_features, train_labels,
                    epochs=30,
                    batch_size=20,
                    validation_data=(validation_features, validation_labels))

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

# =========================
# Pruebo con los originales
def cargar_imagen(im_path):
    img = image.load_img(im_path, target_size=im_size, grayscale=True)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return x

train_generator = datagen.flow_from_directory(train_dir, **generator_params)
categorias = {k:v for v, k in train_generator.class_indices.items()}
imgs_paths = contenidos(ori_dir)
for path in imgs_paths:
	x = cargar_imagen(path)
	preds = model.predict(x)
	preds = np.squeeze(preds)
	print(os.path.basename(path))
	print('{}: {:.0f}% \t {}: {:.0f}%'.format(categorias[0], preds[0]*100, categorias[1], preds[1]*100))

# =========================
# Testeo para cantos reales
ori_dir_chin = os.path.join('nuevos', 'originales', 'sonos', MODO, 'chingolo')
ori_dir_bent = os.path.join('nuevos', 'originales', 'sonos', MODO, 'benteveo')

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
    print('acc =', np.round(acc, 2), ', err =', np.round(err, 2), file=resultados_out)
    print('recB =', np.round(recB, 2), ', recC =', np.round(recC, 2), file=resultados_out)
    print('precB =', np.round(precB, 2), ', precC =', np.round(precC, 2), file=resultados_out)

np.save(nombre_guardado + '/conf_mat.npy', cm)