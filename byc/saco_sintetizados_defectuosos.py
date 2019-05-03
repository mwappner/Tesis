# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 00:31:05 2019

@author: Marcos
"""

import os
import matplotlib.pyplot as plt
from matplotlib.image import imread
#from scipy.io import wavfile
import numpy as np

#%%

base = 'C:\\Users\\Marcos\\Documents\\Facu\\Tesis\\byc'

#carpeta = os.path.join(base, 'sintetizados', 'sonogramas', 'train', 'Chingolos')
#carpeta = os.path.join(base, 'sintetizados', 'sonogramas', 'train', 'Benteveos')
#carpeta = os.path.join(base, 'sintetizados', 'sonogramas', 'validation', 'Chingolos')
carpeta = os.path.join(base, 'sintetizados', 'sonogramas', 'validation', 'Benteveos')

archivos = [os.path.join(carpeta, a) for a in os.listdir(carpeta)]

medias = []
for a in archivos:
    medias.append(np.mean(imread(a)))
    
plt.hist(medias)
#%%

corte = 200

sacar = [a for a, m in zip(archivos, medias) if m>corte]

print('Quedan {} archivos.'.format(len(archivos)-len(sacar)))

#CHEQUEAR!
#%%
for s in sacar:
    os.remove(s)