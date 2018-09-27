# -*- coding: utf-8 -*-
"""
Para visualizar activación de cada filtro dada una imagen

"""
import os
import numpy as np
from keras.preprocessing import image
from keras.datasets import mnist
from keras.models import load_model

#%% cambio el directorio y cargo modelo

os.chdir('/home/marcos/Documents/Probando codigos')

modelo = load_model('MNIST_entero.h5')
modelo.summary() #recuerdo qué tenía

#%% cargo el dataset y me quedo con una de las de entrenamiento

k = 126 #elijo qué imagen mirar

(imagenes, labels) = mnist.load_data()[1] #cargo sólo las imagenes de prueba

#preproceso igual que cuando entrené la red
imagenes = imagenes.reshape((10000, 28, 28, 1)) #le agrego una columna
imagenes= imagenes.astype('float32') / 255 #lo hago un double en [0,1]

import matplotlib.pyplot as plt

img_tensor = image.img_to_array(imagenes[k,:,:])
img_tensor = np.expand_dims(img_tensor, axis=0)

plt.matshow(np.squeeze(img_tensor), cmap='viridis') #squeeze para que me deje plotear algo en grises
#plt.imshow(np.squeeze(img_tensor,axis=3)[0]) #squeeze para que me deje plotear algo en grises
#plt.show()
print('label: {}'.format(labels[k]))

#%% Creo un modelo para mirar los filtros

from keras import models

#Este modelo toma como input una imagen y da como output lo que cada capa escupe
layer_outputs = [layer.output for layer in modelo.layers[:5]] #cargo las capas convolucionales
activation_model = models.Model(inputs=modelo.input, outputs=layer_outputs) #creo el modelo

#miro alguna activación
activations = activation_model.predict(img_tensor)
first_layer_activation = activations[0] #primera capa
print(first_layer_activation.shape) #pinta de los filtros de la capa
plt.matshow(first_layer_activation[0, :, :, 4], cmap='viridis') #quito filtro

#%% Grafico todas las activaciones de todas las capas dada una imagen

#nombres de las capas
layer_names = []
for layer in modelo.layers[:5]: layer_names.append(layer.name)

im_por_fila = 8

def bitificar8(im,desv=1):
 #normalizo la imagen            
    im -= im.mean()
    im /= (im.std() + 1e-5) #por si tengo cero
    im *= desv
#la llevo a 8 bits
    im *= 64
    im += 128
    im = np.clip(im, 0, 255).astype('uint8') #corta todo por fuera de la escala
    return im

for layer_name, layer_activation in zip(layer_names, activations): #zip simplemente me da dos iteradores
    n_features = layer_activation.shape[-1] #cant de filtros
    size = layer_activation.shape[1] #tamaño del filtro de la capa (es cuadrado)
    size += 1 #para agregar bordes

    n_cols = n_features // im_por_fila #cant de columnas (con floor--> descarta sobrantes?)
    display_grid = np.zeros((size * n_cols + 1 , im_por_fila * size + 1)) #hago la cuadricula de filtros

    for col in range(n_cols):
        for row in range(im_por_fila):
            channel_image = layer_activation[0,:, :,col * im_por_fila + row]
        #convierto en 8bits
            channel_image = bitificar8(channel_image,2)
        #agrego ceros para separar
            channel_image = np.pad(channel_image,((1,0),(1,0)),'constant',constant_values=0)
        #lleno la grilla
            display_grid[col * size : (col + 1) * size,
                         row * size : (row + 1) * size] = channel_image
                             
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
    scale * display_grid.shape[0]))
    plt.title('{0}: {1}x{1}'.format(layer_name,size-1))
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis',interpolation='nearest')

    