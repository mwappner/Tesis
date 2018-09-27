# -*- coding: utf-8 -*-
"""
Visualizo filtros

"""

import os
import numpy as np
from keras.models import load_model

import matplotlib.pyplot as plt

#%% cambio el directorio y cargo modelo

os.chdir('/home/marcos/Documents/Probando codigos')

modelo = load_model('MNIST_entero.h5')
modelo.summary() #recuerdo qué tenía


    
#%% Defino funciones útiles
    
from keras import backend as K

#devuelve imagen en 8bits, puede rescalar la desv
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

#cre una func. que arma la imagen visualizada
def generate_pattern(layer_name, filter_index, size=28,canales=1,iteraciones=40):
    
    layer_output = modelo.get_layer(layer_name).output #la capa
    loss = K.mean(layer_output[:, :, :, filter_index]) #algo que maximiza con la capa
    grads = K.gradients(loss, modelo.input)[0] #gradiente de esto respecto a un input
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5) #normalizo y sumo un poquito para no dividir por cero
    iterate = K.function([modelo.input], [loss, grads])
    input_img_data = np.random.random((1, size, size, canales)) * 20 + 128. #la imagen arranca siendo gris ruidoso
    
    step = 1. #tamaño del avance del gradiente
    for i in range(iteraciones): #itero cuarenta veces
        loss_value, grads_value = iterate([input_img_data]) 
        input_img_data += grads_value * step #nuevo input corrigiendo con el gradiente

    img = input_img_data[0]
    return bitificar8(img,0.1)

#pasa de índice lineal a matricial   
def filcol(ind,ancho):

    fil = ind // ancho
    col = ind % ancho
    return fil,col


#%% Grafico filtros
cant_capas_conv = 5 #hasta qué capa del modelo miro (las convloucionales)
size = 28 #tamaño dela imagen (espero cuadradas, si no tendré que reescribir)
size += 1 #para agregar bordes
im_por_fila = 8

for l in range(cant_capas_conv): #zip simplemente me da dos iteradores
    nombre = modelo.layers[l].name
    
    print('Corriendo capa {}: {}'.format(l,nombre))
    
    #salteo el paso si no es una capa con filtros (o sea, max_pooling):
    if nombre[:3] == 'max': 
        print('Salteado.')
        continue
    
    cant_filtros = modelo.layers[l].output_shape[3] #cant de filtros
    n_cols = cant_filtros // im_por_fila #cant de columnas (con floor--> descarta sobrantes?)
    display_grid = np.full((size  * n_cols + 1 , im_por_fila * size + 1),np.nan) #hago la cuadricula de filtros

    for filtro in range(cant_filtros):
        print('Filtro {} de {}'.format(filtro,cant_filtros))
    #muestro filtro
        channel_image = generate_pattern(nombre,filtro)
        channel_image = np.squeeze(channel_image)
    #agrego ceros para poner bordes
#        channel_image = np.pad(channel_image,((1,0),(1,0)),'constant',constant_values=np.nan)
    #lleno la grilla
        row,col = filcol(filtro,n_cols)
        display_grid[col * size + 1: (col + 1) * size,
                     row * size + 1: (row + 1) * size] = channel_image
                             
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
    scale * display_grid.shape[0]))
    plt.title('{0}: {1}x{1}'.format(nombre,size-1))
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis',interpolation='nearest')
    