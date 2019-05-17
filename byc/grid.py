# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 22:05:33 2019

@author: Marcos
"""

import numpy as np
import matplotlib.pyplot as plt
#%%

class Grid:
    '''Una clase para crear y llenar una grilla con imagenes.'''
       
    def __init__(self, cant, fill=np.nan, trasponer=False, bordes=True):

        self.cant = cant #cantidad de imagenes
        self.trasponer = trasponer #por default, la grilla es más ancha que alta
        self.bordes = bordes #si debe o no haber un margen entre figus.
        self.shape = self._cant_to_mat() #tamaño de la matriz de imagenes

        self.grid = None #la grilla a llenar con imagenes
        #self.im_shape = None #tamaño de la imagen
        self.ind = 0 #por qué imagen voy?
        self.fill_with = fill #con qué lleno la grilla vacía

    @property
    def im_shape(self):
        return self._im_shape_real
    @im_shape.setter
    def im_shape(self, value):
        self._im_shape_real = value
        self._imRGB = len(value)==3 #if image is RGB
        
        if self.bordes:
            self._im_shape_bordes = (value[0] + 1, value[1] + 1)
        else:
            self._im_shape_bordes = self.im_shape

    def _cant_to_mat(self):
        '''Dimensiones de la cuadrícula más pequeña y más cuadrada
        posible que puede albergar [self.cant] cosas.'''
        col = int(np.ceil(np.sqrt(cant)))
        row = int(round(np.sqrt(cant)))
        if self.trasponer:
            return col, row
        else:
            return row, col
        
    def _filcol(self):
        '''Pasa de índice lineal a matricial.'''
        fil = self.ind // self.shape[1]
        col = self.ind % self.shape[1]
        return int(fil), int(col)

    def _create_grid(self):
        shape = (self._im_shape_bordes[0] * self.shape[0], 
                 self._im_shape_bordes[1] * self.shape[1])
        if self.bordes:
            shape = shape[0] + 1, shape[1] + 1
        if self._imRGB:
            shape = *shape, 3
            
        self.grid = np.full(shape, self.fill_with)

    def insert_image(self, im):
        '''Agrego una imagen a la grilla.'''
        #inicializo la grilla
        if self.grid is None:
            self.im_shape = im.shape
            self._create_grid()

        #la lleno
        col, row = self._filcol()
        #sumo el booleano de bordes para que cierre bien la cuenta
        if self._imRGB:
            if im.ndim != 3:
                raise ValueError('Imagen debe ser RGB (tres canales).')
            self.grid[col * self._im_shape_bordes[0] + int(self.bordes) : 
                        (col + 1) * self._im_shape_bordes[0],
                    row * self._im_shape_bordes[1] + int(self.bordes) : 
                        (row + 1) * self._im_shape_bordes[1], :]= im
        else:
            if im.ndim != 2:
                raise ValueError('Imagen debe ser escala de ggrises (un canal).')     
            self.grid[col * self._im_shape_bordes[0] + int(self.bordes) : 
                        (col + 1) * self._im_shape_bordes[0],
                    row * self._im_shape_bordes[1] + int(self.bordes) : 
                        (row + 1) * self._im_shape_bordes[1]]= im
        
        #avanzo el contador apra la siguiente imagen
        self.ind += 1

        
#%%
        
cant = 33
shape = (21,25)
g = Grid(cant, trasponer=False, bordes=True, fill=100)
for i in range(cant):
    g.insert_image(np.ones(shape)*i)

plt.matshow(g.grid)
plt.grid()

#%%

cant = 17
shape = (11,9)
g = Grid(cant, trasponer=False, bordes=True, fill=np.nan)
colores = [(0,0,0), (1,0,0), (0,1,0), (0,0,1), (0,1,1), (1,0,1), (1,1,0), 
           (1,1,1), (.5,.5,.5), (1,.5,0), (1,0,.5), (.5,1,0), (.5,0,1),
           (0,.5,1), (0,1,.5), (.5,0,0), (0,.5,0), (0,0,.5)]
imagenes = []
for c in colores:
    liso = np.ones((*shape,3))
    for i in range(3):
        liso[:,:,i] *= c[i]
    imagenes.append(liso)

for i in range(cant):
    g.insert_image(imagenes[i])

plt.imshow(g.grid)
plt.grid()
