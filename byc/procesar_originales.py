# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 02:22:12 2019

@author: Marcos
"""

import os
#from shutil import copyfile
from utils import FiltroSonograma, contenidos

modos = 'pad', 'stretch', 'center'

bent_original = 'nuevos/originales/benteveo_BVRoRo_highpass_notch.wav'
ching_original = 'nuevos/originales/chingolo_XC462515_denoised.wav'
#carpeta_de_llegada_ori = lambda modo: os.path.join('sintetizados', 'dnn', modo, 'originales')
carpeta_de_llegada = lambda modo: os.path.join('nuevos', 'originales', 'sonos', modo)

def procesar_benteveo(archivo, destino, dur=None, modo='pad'):
    s = FiltroSonograma(archivo, target_duration=dur)

    s.rango_dinamico(0.5)
    s.thresholdear()
    s.bitificar8(desv=2, ceros=False)

    # Si no hago ninguno de estos, estoy estirando 
    if modo=='pad':
    	s.cut_or_extend(centered=False)
    elif modo=='center':
    	s.cut_or_extend(centered=True)

    s.guardar(ubicacion=destino)
    
def procesar_chingolo(archivo, destino, dur=None):
    s = FiltroSonograma(archivo, target_duration=dur)

    s.rango_dinamico(0.5)
    s.thresholdear()
    s.cut_or_extend()
    s.guardar(ubicacion=destino)

def mover(origen, destino, cant, subcarpeta='train'):
    '''los rachivos en la carpeta train de origen se van a destino (validate o test) correspondiente.'''
    for file in contenidos(origen)[-cant:]:
        destino = file.replace(subcarpeta, destino)
        os.rename(file, destino)
    
originales = contenidos('nuevos/originales', filter_ext='.wav')
originales.print_orden()

DURACION = 1.8 # tomado de la síntesis de chingolos, el más largo de los dos
#%%

elegidos = [2, 7, 3, 5]

for k in elegidos:
    sonido = originales[k]
    if 'benteveo' in sonido:
        procesar_benteveo(sonido, carpeta_de_llegada(''), dur=DURACION)
        
    if 'chingolo' in sonido:
        procesar_chingolo(sonido, carpeta_de_llegada(''), dur=DURACION)