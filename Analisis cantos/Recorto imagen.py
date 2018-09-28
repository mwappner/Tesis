# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 12:06:27 2018

@author: Marcos
"""

from scipy.io import wavfile
import numpy as np
import matplotlib
matplotlib.use("Qt5Agg") # This program works with Qt only
import pylab as pl
from PIL import Image
import os
#from skimage import filters #para otsu

motivoPath = os.path.join(os.getcwd(),'Motivos')
sonoPath = os.path.join(motivoPath,'Sonogramas','Nuevos')
sonoFiles = os.listdir(sonoPath)
sonoFiles.sort()

with open(os.path.join(motivoPath,'duraciones_nuevos.txt'),'r') as d:
    c_d = d.readlines()
#diccionario que tiene numbre_del_archivo:duracion
#duración en float y al nombre le saco el '\n' final
dur = {k.strip():float(v) for v,k in (s.split(', ') for s in c_d)}

#%%
#devuelve el ancho de la ventana actual
def Ancho(axes):
    xa = axes.get_xlim()
    return xa[1]-xa[0]

#seteo el nuevo ancho como 
def NuevoAncho(axes):
    ancho = Ancho(axes)
    xla = axes.set_xlabel('Ancho: {:.3f}s'.format(ancho),fontsize=18,fontweight='bold')
    return xla

#%% Importo y trabajo sonogramas


escala = 2500 #cantidad de pixeles del sonograma por segundo
cual = 0 #qué figura miro
nombre_actual = sonoFiles[cual][:-4]

#cargo imagen, transformo a escala de grises ('L')
sono1 = Image.open(sonoPath + '/' + sonoFiles[cual]).convert('L')
#recorto la parte que me interesa (dentro de los ejes)
sono1 = sono1.crop((360, 238, 6845, 1860)) #obtenido a mano
#reescalo la imagen para que el tamaño refleje la duración
#convierto a np.array
alto = sono1.size[1]
ancho = int(dur[nombre_actual] * escala)
A = np.array(sono1.resize((ancho,alto),Image.LANCZOS))
A = 255-A #la escala está al revés (255 es poca potencia)


#Plots:
fig, ax1 = pl.subplots()
#fig.set_size_inches([18.39,  9.27])

ax1.imshow(A,cmap='Greys',extent=[0, dur[nombre_actual],0, 0.1])
#ax1.plot(tiempo[:-1],diffThreshCu*0.5,'g--')


titulo = '{}: {}, duración:{:.1f}s'.format(cual,nombre_actual,dur[nombre_actual])
ax1.set_title(titulo)
xla = NuevoAncho(ax1)
#%%
def cortar(im, ti, dur, fi, ff):
    ic = im[int(ff*alto * 10) : int(fi*alto * 10), int(ti*escala) : int((ti+dur) * escala)]
    return ic

#ic = A[0:alto, int(0.27*escala): int((0.27 + 0.05) * escala)]
ic = cortar(A, 0.08, 0.09, 0.055, 0.031)
pl.imshow(ic)