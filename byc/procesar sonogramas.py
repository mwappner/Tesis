# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 17:46:43 2019

@author: Marcos
"""
import os
from filtro_sonogramas import FiltroSonograma

carpeta_de_salida_ching = os.path.join('sintetizados', 'locales', 'audios', 'chingolos')
carpeta_de_llegada_ching = os.path.join('sintetizados', 'locales', 'sonogramas', 'chingolos')

carpeta_de_salida_bent = os.path.join('sintetizados', 'audios')
carpeta_de_llegada_bent = os.path.join('sintetizados', 'procesados')

archivos = lambda direc: [os.path.join(direc, f) for f in os.listdir(direc)]

def procesar_benteveo(archivo, dur=None):
    s = FiltroSonograma(archivo, target_duration=dur)

    s.rango_dinamico(0.5)
    s.thresholdear()
    s.bitificar8(desv=2, ceros=False)
    s.cut_or_extend(centered=False)
    s.guardar(ubicacion=carpeta_de_llegada_bent)
    
def procesar_chingolo(archivo, dur=None):
    s = FiltroSonograma(archivo, target_duration=dur)

    s.rango_dinamico(0.5)
    s.thresholdear()
    s.cut_or_extend()
    s.guardar(ubicacion=carpeta_de_llegada_ching)
    
#%%

DURACION = 1.8 # tomado de la síntesis de chingolos, el más largo de los dos

for file in archivos(carpeta_de_salida_ching):
    procesar_chingolo(file, dur=DURACION)
    
for file in archivos(carpeta_de_salida_bent):
    if 'benteveo' not in file:
        continue
    procesar_benteveo(file, dur=DURACION)