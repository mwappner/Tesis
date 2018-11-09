# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 12:26:24 2018

@author: Marcos
"""


import numpy as np
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

escala = 2500 #cantidad de pixeles del sonograma por segundo


#%%

def abro_imagen(file):
    
    #recupero sólo el nombre sin extensión ni path
    nombre_actual = os.path.basename(file)[:-4]
    #cargo imagen, transformo a escala de grises ('L')
    sono1 = Image.open(file).convert('L')
    #recorto la parte que me interesa (dentro de los ejes)
    sono1 = sono1.crop((360, 238, 6845, 1860)) #obtenido a mano
    #reescalo la imagen para que el tamaño refleje la duración
    #convierto a np.array
    
    global alto
    alto = sono1.size[1]
    
    ancho = int(dur[nombre_actual] * escala)
    A = np.array(sono1.resize((ancho,alto),Image.LANCZOS))
    A = 255-A #la escala está al revés (255 es poca potencia)
    return A

def cortar(im, ti, dur, fi, ff):
    #por si pongo las frecuencias al revés
    if fi>ff:
        fi, ff = ff, fi
    #recorto:
    ic = im[int(fi*alto * 10) : int(ff*alto * 10), int(ti*escala) : int((ti+dur) * escala)]
    return ic

def guardar(im_array, nombre, extension='png'):
    ''' Toma un array que contiene una imagen y un nombre (path completo,
    sin extensión). Opcionalmente una extensión de archivo (default: 'png')'''
    im = Image.fromarray(im_array)
    nombre = new_file(nombre + '.' + extension)
    im.save(nombre)

#ic = A[0:alto, int(0.27*escala): int((0.27 + 0.05) * escala)]
#ic = cortar(A, 0.08, 0.09, 0.055, 0.031)
#pl.imshow(ic)

#%%

def new_file(name, newseparator='_'):
    '''Returns a name of a unique file or directory so as to not overwrite.
    If propsed name existed, will return name + newseparator + number.
     
    Parameters:
    -----------
        name : str (path)
            proposed file or directory name influding file extension
        nweseparator : str
            separator between original name and index that gives unique name
    '''
    
    #if file is a directory, extension will be empty
    base, extension = os.path.splitext(name)
    i = 2
    while os.path.exists(name):
        name = base + newseparator + str(i) + extension
        i += 1
        
    return name
