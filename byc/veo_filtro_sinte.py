# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 22:34:53 2019

@author: Marcos
"""

from matplotlib import pyplot as plt
import os
from scipy.io import wavfile
from numpy import zeros, arange, where


fsamp = 882000.0
tiempo_total = 0.86
cant_puntos = int(tiempo_total*fsamp)

def rectas(ti, tf, wi, wf, freqs):
        
    i=int(ti*fsamp)
    j=int(tf*fsamp)
    dj = j-i
    k = arange(dj) / dj # =  t - ti / (ti - tf)
        
    freqs[i:j] = wi + (wf-wi) * k


#%%
carpeta = 'filtro'
lista = [os.path.join(carpeta,f) for f in os.listdir(carpeta) if f.endswith('wav')]

frecs = zeros(cant_puntos)
rectas(0.01, tiempo_total-0.01, 200, 5000, frecs)
frecs = frecs[::20]
donde = where(frecs>0)[::10]

#%%

fig, axarr  = plt.subplots(2,2)

for f, ax in zip(lista, axarr.flatten()):
    _, signal = wavfile.read(f)
    ax.plot(frecs[donde],signal[donde])
    ax.set_title(f)