# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 02:51:17 2019

@author: Marcos
"""

import os
import numpy as np
from scipy.signal import spectrogram
from scipy.io import wavfile
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

from skimage import filters


#%%

class FiltroSonograma:
    
    def __init__(self, archivo, target_duration=None, limites_frec=(10, 8000), *args, **kwargs):
        self.nombre = archivo
        self.lims = limites_frec
        self.target_dur = target_duration
        
        self.fs, self.sonido = wavfile.read(archivo)
        self._hago_sonograma(sigma=.15, *args, **kwargs)

    def _hago_sonograma(self, dur_seg=0.012, overlap=.9, sigma=.25, 
                       gauss_filt={'sigma':0.1}):
        '''Calcula un sonograma con ventana gaussiana. Overlap y sigma son 
        proporcionales al tamaño de la ventana. Tamaño de la ventana dado en 
        segundos. Devuelve un sonograma en escala logarítmica.
        Por defecto aplica un filtro gaussiano de orden 1 a la salida usando 
        sigma=0.1. Para no aplicarlo, fijar gauss_filt a diccionario vacío o a
        False.'''
        if not 0<=overlap<=1 or not 0<=sigma<=1:
            raise ValueError('overlap y sigma deben estar entre 0 y 1.')
        
        nperseg = int(dur_seg*self.fs)
        f, t, sxx = spectrogram(self.sonido, fs=self.fs,
                           window=('gaussian',nperseg*sigma),
                           nperseg=nperseg,
                           noverlap=int(nperseg*overlap),
                           scaling='spectrum')
        
        sxx = sxx[np.logical_and(f>self.lims[0], f<self.lims[1]), :]
        self.frecuencias = f[np.logical_and(f>self.lims[0], f<self.lims[1])]
        self.tiempos = t
        self.target_dur = self.target_dur or t[-1]
        
        if gauss_filt:
            self.sono = np.log10(gaussian_filter(sxx + 1e-6, sigma=1))
        else:
            self.sono = np.log10(sxx + 1e-6)
        
        
    def bitificar8(self, desv=1, ceros=True):
        '''Devuelve la imagen pasada a 8 bits. Puede reescalar la desviación.
        La variable ceros define si se deben tener en cuenta los ceros en la 
        imagen a la hora de calcular media y desviación.'''
        #normalizo la imagen a [-1, 1]
        if self.sono.dtype == np.float32:
            im = self.sono
        else:
            im = self.sono.astype(np.float32)
        #Decido si contar o no los ceros en el cálculo d la media
        if ceros:        
            im -= im.mean()
        else:
            im -= im[im>0].mean()
            
        im /= (im.std() + 1e-5) #por si tengo cero
        im *= desv
        #la llevo a 8 bits
        im *= 64
        im += 128
        im = np.clip(im, 0, 255).astype('uint8') #corta todo por fuera de la escala
        self.sono = im
    
    
    def normalizar(self, bits=True):
        '''Devuelve una copia del sonograma rescalado para ocupat todo el rango
        dinámico del formato correspondiente. Si bits=True, devuelve una imagen
        en 8 bits. Si no, la escala será [0,1].'''
        
        im = self.sono
        
        im -= im.min()
        im /= im.max() + 1e-10 #por si tengo cero
        if not bits:
            return im
        im *= 255
        self.sono = im.astype('uint8')
    
    
    def thresholdear(self):
        '''Filtra una imagen utilziando el criterio de otsu.'''
        im = self.sono
        self.sono = np.where(im>filters.threshold_otsu(im), im, 0)
    
    
    def rango_dinamico(self, valor, bits=True):
        '''Escala la imagen de forma que todos los valores que estén un porcentaje 
        <valor> debajo del máximo pasan a cero y el máximo pasa a ser 1 o 255,
        dependiendo de si bits=True o False, respecivamente.'''
        imagen = self.sono

        if not 0<=valor<=1:
            raise ValueError('valor debe ser entre 0 y 1')
        self.normalizar(bits=False) #valores ahora en [0,1]
        imagen[imagen<valor] = valor
        self.normalizar(bits)
    
    
    def cut_or_extend(self):
        '''Si la duración actual del sonido es más grande que la del sonido 
        objetivo, recorta el final. Si es más chica, rellena con ceros.'''
        if self.tiempos[-1] > self.target_dur:
            self.sono = self.sono[:, self.tiempos<self.target_dur]
        elif self.tiempos[-1] < self.target_dur:
            dt = self.tiempos[1] - self.tiempos[0]
            cant_faltante = int((self.target_dur - self.tiempos[-1])/dt)
            self.sono = np.append(self.sono, 
                                  np.zeros((self.sono.shape[0], cant_faltante)), 
                                  axis=1)
        #redefino el vector de tiempos para graficar 
        self.tiempos = np.linspace(self.tiempos[0], self.target_dur, self.sono.shape[1])
            
    
    def plotear(self, im=None, ax=None, log=False, labels=False):
        
        im = im or self.sono #si no le di una imagen, uso el guardado
        
        if ax is None:
            fig, ax = plt.subplots()
        if log:
            im = np.log10(im)
        ax.pcolormesh(self.tiempos, self.frecuencias/1000, im,
                      rasterized=True, cmap=plt.get_cmap('Greys'))
        if labels:
            plt.xlabel('tiempo [s]')
            plt.ylabel('frecuencia [Hz]')
            plt.title(self.archivo, fontsize=15)
