# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 01:55:39 2019

@author: Marcos
"""

import os

import numpy as np

from keras.models import load_model
from keras.preprocessing import image
#from keras.preprocessing.image import ImageDataGenerator

from utils import contenidos

model_dir = 'modelos'
modelos = contenidos(model_dir)
modelos.print_orden()

modo = 'pad'
ori_dir_chin = os.path.join('nuevos', 'originales', 'sonos', modo, 'chingolo')
ori_dir_bent = os.path.join('nuevos', 'originales', 'sonos', modo, 'benteveo')


def cargar_imagen(im_path):
    img = image.load_img(im_path, target_size=im_size, grayscale=True)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return x

#%%

im_size = (300, 200)

este_modelo = 1
model = load_model(contenidos(modelos[este_modelo], filter_ext='.h6')[0])

categorias = {0:'bent', 1:'ching'}
#imgs_paths = contenidos(ori_dir_bent)
#imgs_paths = contenidos(ori_dir_chin)
resultados = {'bent':[], 'ching':[]}
paths = {'bent':ori_dir_bent, 'ching':ori_dir_chin}

for pajaro in paths:
    print(pajaro.upper())
    print()
    for i, path in enumerate(contenidos(paths[pajaro])):
        
        x = cargar_imagen(path)
        
        preds = model.predict(x)
        preds = np.squeeze(preds)
        
        resultados[pajaro].append(preds)
        print(i, '{}: {:.0f}% \t {}: {:.0f}%'.format(categorias[0], preds[0]*100, categorias[1], preds[1]*100))
    print()

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

print(cm)
print('acc =', np.round(acc, 2), ', err =', np.round(err, 2))
print('recB =', np.round(recB, 2), ', recC =', np.round(recC, 2))
print('precB =', np.round(precB, 2), ', precC =', np.round(precC, 2))