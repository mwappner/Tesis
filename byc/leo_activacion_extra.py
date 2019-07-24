# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 02:08:06 2019

@author: Marcos
"""

import numpy as np
import matplotlib.pyplot as plt

from utils import Grid

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
#%%

activations = np.load('activations.npz')
activations = [v for v in activations.values()]

for layer_activation in activations: #zip simplemente me da dos iteradores
        
    n_features = layer_activation.shape[-1] #cant de filtros
    size = layer_activation.shape[1] #tamaño del filtro de la capa (es cuadrado)

    g = Grid(n_features, fill_with=0)    
    for ch_image in range(n_features):
        channel_image = layer_activation[0,:, :, ch_image]
        g.insert_image(bitificar8(channel_image, 2))

    g.show()
#    plt.title('{0}: {1}x{1}'.format(layer_name,size-1))
    plt.grid(False)