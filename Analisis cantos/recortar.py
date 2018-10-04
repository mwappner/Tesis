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
    
    #sólo el nombre sin extensión ni path
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

def guardar(im_array, nombre):    
    im = Image.fromarray(im_array)
    nombre = free_file(nombre + '.png')
    im.save(nombre)

#ic = A[0:alto, int(0.27*escala): int((0.27 + 0.05) * escala)]
#ic = cortar(A, 0.08, 0.09, 0.055, 0.031)
#pl.imshow(ic)

#%%

def new_dir(my_dir, newformat='{}_{}'):
    
    """Makes and returns a new directory to avoid overwriting.
    
    Takes a directory name 'my_dir' and checks whether it already 
    exists. If it doesn't, it returns 'dirname'. If it does, it 
    returns a related unoccupied directory name. In both cases, 
    the returned directory is initialized.
    
    Parameters
    ----------
    my_dir : str
        Desired directory (should also contain full path).
    
    Returns
    -------
    new_dir : str
        New directory (contains full path)
    
    Yields
    ------
    new_dir : directory
    
    """
    
    sepformat = newformat.split('{}')
    base = os.path.split(my_dir)[0]
    
    new_dir = my_dir
    while os.path.isdir(new_dir):
        new_dir = os.path.basename(new_dir)
        new_dir = new_dir.split(sepformat[-2])[-1]
        try:
            new_dir = new_dir.split(sepformat[-1])[0]
        except ValueError:
            new_dir = new_dir
        try:
            new_dir = newformat.format(my_dir, str(int(new_dir)+1))
        except ValueError:
            new_dir = newformat.format(my_dir, 2)
        new_dir = os.path.join(base, new_dir)
    os.makedirs(new_dir)
        
    return new_dir

#%%

def free_file(my_file, newformat='{}_{}'):
    
    """Returns a name for a new file to avoid overwriting.
        
    Takes a file name 'my_file'. It returns a related unnocupied 
    file name 'free_file'. If necessary, it makes a new 
    directory to agree with 'my_file' path.
        
    Parameters
    ----------
    my_file : str
        Tentative file name (must contain full path and extension).
    newformat='{}_{}' : str
        Format string that indicates how to make new names.
    
    Returns
    -------
    new_fname : str
        Unoccupied file name (also contains full path and extension).
        
    """
    
    base = os.path.split(my_file)[0]
    extension = os.path.splitext(my_file)[-1]
    
    if not os.path.isdir(base):
        os.makedirs(base)
        free_file = my_file
    
    else:
        sepformat = newformat.split('{}')
        free_file = my_file
        while os.path.isfile(free_file):
            free_file = os.path.splitext(free_file)[0]
            free_file = free_file.split(sepformat[-2])[-1]
            try:
                free_file = free_file.split(sepformat[-1])[0]
            except ValueError:
                free_file = free_file
            try:
                free_file = newformat.format(
                        os.path.splitext(my_file)[1],
                        str(int(free_file)+1),
                        )
            except ValueError:
                free_file = newformat.format(free_file, 2)
            free_file = os.path.join(base, free_file+extension)
    
    return free_file
