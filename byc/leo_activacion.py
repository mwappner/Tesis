# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 01:03:25 2019

@author: Marcos
"""

import os
import numpy as np

import matplotlib.pyplot as plt

from keras import models
from keras.preprocessing import image

from utils import contenidos, Grid, new_name

modo = 'pad'

train_dir_bent = os.path.join('sintetizados','dnn', modo, 'train', 'benteveo')
train_dir_chin = os.path.join('sintetizados','dnn', modo, 'train', 'chingolo')

ori_dir_chin = os.path.join('nuevos', 'originales', 'sonos', modo, 'chingolo')
ori_dir_bent = os.path.join('nuevos', 'originales', 'sonos', modo, 'benteveo')

model_dir = 'modelos'
modelos = contenidos(model_dir)
modelos.print_orden()


im_size = (300, 200) 
def cargar_imagen(im_path):
    img = image.load_img(im_path, target_size=im_size, color_mode = "grayscale")
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return x

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
#%% Elijo modelo y corro

este_modelo = 0
modelo = models.load_model(contenidos(modelos[este_modelo], filter_ext='.h6')[0])

########MODIFICAR ESTO###########
#imagen_nombres = contenidos(ori_dir_chin)
#imagenes = 2, 6, 9, 11, 21, 22, 26 #chingolo
#carpeta_de_guardado_base = 'ori_chin' #chingolo

imagen_nombres = contenidos(ori_dir_bent)
imagenes = 1, 3, 9, 23, 24 #benteveo
carpeta_de_guardado_base = 'ori_bent' #benteveo
################################

for i in imagenes:
    print('Corriendo', i)
    imagen_nombre = imagen_nombres[i]
    
    carpeta_de_guardado = '_'.join([carpeta_de_guardado_base ,str(i), os.path.basename(imagen_nombre).split('-')[0][:-1]])
    carpeta_de_guardado = new_name(os.path.join(modelos[este_modelo], carpeta_de_guardado))
    os.makedirs(carpeta_de_guardado)
    
    una_imagen = cargar_imagen(imagen_nombre)
    
    ### Creo un modelo para mirar activaciones ###
    
    #Este modelo toma como input una imagen y da como output lo que cada capa escupe
    #layer_outputs = [layer.output for layer in modelo.layers[:5]] #cargo las capas convolucionales
    layer_outputs = [layer.output for layer in modelo.layers if 'conv' in layer.name] #cargo las capas convolucionales
    activation_model = models.Model(inputs=modelo.input, outputs=layer_outputs) #creo el modelo
    
    #miro alguna activación
    activations = activation_model.predict(una_imagen)
    #first_layer_activation = activations[0] #primera capa
    #print(first_layer_activation.shape) #pinta de los filtros de la capa
    #plt.matshow(first_layer_activation[0, :, :, 0], cmap='viridis') #quito filtro
    
    layer_names = [l.name for l in modelo.layers if 'conv' in l.name]
    
    for layer_name, layer_activation in zip(layer_names, activations): #zip simplemente me da dos iteradores
            
        n_features = layer_activation.shape[-1] #cant de filtros
        size = layer_activation.shape[1:3] #tamaño del filtro de la capa (es cuadrado)
    
        g = Grid(n_features, fill_with=np.nan, bordes=True)    
        for ch_image in range(n_features):
            channel_image = layer_activation[0,:, :, ch_image]
            g.insert_image(bitificar8(channel_image, 2))
    
        g.show()
        plt.title('{}: {}x{}'.format(layer_name,*size))
        plt.grid(False)
        plt.savefig(os.path.join(carpeta_de_guardado, layer_name + '.jpg'), bbox='tight', dpi=400)
        plt.close()
        
print('\a')
