# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 01:55:39 2019

@author: Marcos
"""

import os

import numpy as np

from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

from utils import contenidos

model_dir = 'modelos'
modelos = contenidos(model_dir)
modelos.print_orden()

ori_dir = os.path.join('nuevos', 'originales', 'sonos')

#%%

im_size = (300, 200)

este_modelo = 3
model = load_model(contenidos(modelos[este_modelo], filter_ext='.h6')[0])

categorias = {0:'bent', 1:'ching'}
imgs_paths = contenidos(ori_dir)
for path in imgs_paths:
	img = image.load_img(path, target_size=im_size, grayscale=True)
	x = image.img_to_array(img)
	x = np.expand_dims(x,axis=0)
	preds = model.predict(x)
	preds = np.squeeze(preds)
	print(os.path.basename(path))
	print('{}: {:.0f}% \t {}: {:.0f}%'.format(categorias[0], preds[0]*100, categorias[1], preds[1]*100))