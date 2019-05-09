# -*- coding: utf-8 -*-
"""
Created on Wed May  1 17:40:15 2019

@author: Marcos
"""

import os
import numpy as np
from scipy.signal import spectrogram
from scipy.io import wavfile
import matplotlib.pyplot as plt

from skimage import filters

from scipy.ndimage import gaussian_filter

archivos = lambda direc: [os.path.join(direc, f) for f in os.listdir(direc)]

def bitificar8(im,desv=1, ceros=True):
    '''Devuelve la imagen pasada a 8 bits. Puede reescalar la desviación.'''
    #normalizo la imagen a [-1, 1]
    
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
    return im

def normalizar(im, bits=True):
    im -= im.min()
    im /= im.max() + 1e-10 #por si tengo cero
    if not bits:
        return im
    im *= 255
    return im.astype('uint8')

thresholdear = lambda im: np.where(im>filters.threshold_otsu(im), im, 0)

def rango_dinamico(valor, imagen, bits=True):
    '''Escala la imagen de forma que todos los valores que estén un porcentaje 
    <valor> debajo del máximo pasan a cero y el máximo pasa a ser 255.'''
    if not 0<=valor<=1:
        raise ValueError('valor debe ser entre 0 y 1')
    imagen = normalizar(imagen, bits=False) #valores ahora en [0,1]
    imagen[imagen<valor] = valor
    return normalizar(imagen, bits)

def mispec(sonido, fs, dur_seg=0.012, overlap=.9, sigma=.25):
    '''Calcula un sonograma con ventana gaussiana. Overlap y sigma son 
    proporcionales al tamaño de la ventana. Tamaño de la ventana dado en 
    segundos.'''
    if not 0<=overlap<=1 or not 0<=sigma<=1:
        raise ValueError('overlap y sigma deben estar entre 0 y 1.')
    
    nperseg = int(dur_seg*fs)
    return spectrogram(sonido, fs=fs,
                       window=('gaussian',nperseg*sigma),
                       nperseg=nperseg,
                       noverlap=int(nperseg*overlap),
                       scaling='spectrum')
#%%
sinte_ubi = os.path.join('limpiando', 'audios') # directorio de los sintetizados
ori_ubi = os.path.join('limpiando', 'originales') # directorio de los originales
sinte = archivos(sinte_ubi)
ori = archivos(ori_ubi)

#calculo sonograma
fs, sonido = wavfile.read(sinte[1])
f, t, Sxx = mispec(sonido, fs, sigma=.15)

#elimino valores correspondientes a frecuencias muy altas
lims = 10, 15000
Sxx = Sxx[np.logical_and(f>lims[0], f<lims[1]), :]
f = f[np.logical_and(f>lims[0], f<lims[1])]

def plotear(im, ax=None, log=False):
    if ax is None:
        fig, ax = plt.subplots()
    if log:
        im = np.log10(im)
    ax.pcolormesh(t, f/1000, im,rasterized=True, cmap=plt.get_cmap('Greys'))
#    ax.ticklabel_format(axis='y', style='sci')
#    ax.set_xlabel('tiempo[s]')
#    ax.set_ylabel('frecuencia [Hz]')
#    ax.axis('off')

##########sin log
#log = True #Para el plot
#s = bitificar8(Sxx)
#so = thresholdear(s)
#
#n = normalizar(Sxx)
#no = thresholdear(n)
#
#sf = gaussian_filter(Sxx, sigma=1)
#sfo = thresholdear(sf) # para s da 150; s.mean() = 155


######con log
log = False #para el plot
s = bitificar8(np.log10(Sxx))
so = thresholdear(s)
soo = bitificar8(so.astype('float'), desv=2, ceros=False)

n = normalizar(np.log10(Sxx))
no = thresholdear(n)
noo = bitificar8(no.astype('float'), desv=2, ceros=False)

sf = normalizar(np.log10(gaussian_filter(Sxx, sigma=1)))
sfo = thresholdear(sf) # para s da 150; s.mean() = 155
sfoo = bitificar8(sfo.astype('float'), desv=2, ceros=False)


fig, axarr = plt.subplots(3,3, sharex=True, sharey=True)
fig.set_size_inches([9.07, 5.98])
ims = (n, no, noo, s, so, soo, sf, sfo, sfoo)
titulos = ('normalizar', 'normalizar otsu', 'normalizar otsu contraste',
           'bitificar',  'bitificar otsu', 'bitificar otsu contraste', 
           'gauss_filt', 'gauss_filt otsu', 'gauss_filt otsu contraste')
for im, ti, ax in zip(ims, titulos, axarr.flatten()):
    plotear(im, ax, log=log)
    ax.set_title(ti)

axarr[1,0].set_ylabel('frecuencia [kHz]')
axarr[2,1].set_xlabel('tiempo [s]')

#%% Pruebo con media por fila

c = 1.15
media_col, media_fil = n.mean(axis=0), n.mean(axis=1)
m = np.where(np.logical_and(n>media_col*c, n>np.expand_dims(media_fil*c, 1)), n, 0)
plotear(m)

#%%
mascarear = lambda c: np.where(np.logical_and(n>media_col*c, n>np.expand_dims(media_fil*c, 1)), n, 0)
k = np.linspace(.8, 4, 500)
medias = [mascarear(val).mean() for val in k]

plt.plot(k, medias/max(medias))
plt.plot(k[1:-1], np.diff(medias, 2))