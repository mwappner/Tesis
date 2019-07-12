# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 17:46:43 2019

@author: Marcos
"""
import os
from shutil import copyfile
from utils import FiltroSonograma, contenidos

modos = 'pad', 'stretch', 'center'

carpeta_de_salida_ching = os.path.join('sintetizados', 'audios', 'chingolos')
carpeta_de_llegada_ching = lambda modo: os.path.join('sintetizados', 'dnn', modo, 'train', 'chingolo')

carpeta_de_salida_bent = os.path.join('sintetizados', 'audios', 'benteveos')
carpeta_de_llegada_bent = lambda modo: os.path.join('sintetizados', 'dnn', modo, 'train', 'benteveo')

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
    
#%%

DURACION = 1.8 # tomado de la síntesis de chingolos, el más largo de los dos

for file in contenidos(carpeta_de_salida_bent)[-100:]:
	for modo in modos: #lo hago en los tres modos
		procesar_benteveo(file, destino=carpeta_de_llegada_bent(modo), dur=DURACION, modo=modo)

for file in contenidos(carpeta_de_salida_ching)[-100:]:
   	procesar_chingolo(file, destino=carpeta_de_llegada_ching('pad'), dur=DURACION)

#copio los chingolos a las carpetas de los tres modos
for modo in modos[1:]:
	for file in contenidos(carpeta_de_llegada_ching('pad')):
		copyfile(file, file.replace('pad', modo))

#muevo a validation
for modo in modos:
	mover(carpeta_de_llegada_bent(modo), destino='validate', cant=500) 
	mover(carpeta_de_llegada_ching(modo), destino='validate', cant=500)

    mover(carpeta_de_llegada_bent(modo), destino='test/test_folder', cant=100, subcarpeta='train/benteveo')
    mover(carpeta_de_llegada_bent(modo), destino='test/test_folder', cant=100, subcarpeta='train/chingolo')