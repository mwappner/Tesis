# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 17:46:43 2019

@author: Marcos
"""
import os
from filtro_sonogramas import FiltroSonograma

archivos = lambda direc: [os.path.join(direc, f) for f in os.listdir(direc)]

def procesar_benteveo(archivo, dur=None):
    s = FiltroSonograma(archivo, target_duration=dur)

    s.rango_dinamico(0.5)
    s.thresholdear()
    s.bitificar8(desv=2, ceros=False)
    #s.cut_or_extend()
    s.plotear()
    
def procesar_chingolo(archivo, dur=None):
    s = FiltroSonograma(archivo, target_duration=dur)

    s.rango_dinamico(0.5)
    s.thresholdear()
    #s.cut_or_extend()
    s.plotear()
#%%
#sinte_ubi = os.path.join('nuevos', 'audios') # directorio de los sintetizados
#ori_ubi = os.path.join('nuevos', 'originales') # directorio de los originales
#sinte = archivos(sinte_ubi)
#ori = archivos(ori_ubi)
#
#for i, f in enumerate(ori):
#    if f.endswith('.wav'):
#        print('ori {}: {}'.format(i,f))
#print('')
#for i, f in enumerate(sinte):
#    if f.endswith('.wav'):
#        print('sinte {}: {}'.format(i,f))

sonidos_ubi = os.path.join('sintetizados', 'locales', 'audios', 'chingolos')
sonidos = archivos(sonidos_ubi)
for i, f in enumerate(sonidos):
    if f.endswith('.wav'):
        print('{}: {}'.format(i,f))
#%%

file_index = 0

#for file_index in range(14):

###====================================###
#Descomentar una línea y comentar el resto
###====================================###
#s = FiltroSonograma(sinte[file_index]) #sonidos sintetizados
#s = FiltroSonograma(ori[file_index]) #sonidos originales
s = FiltroSonograma(sonidos[file_index], gauss_filt=False, target_duration=1.8)

s.rango_dinamico(0.5)
s.thresholdear()
#s.bitificar8(desv=2, ceros=False)
s.cut_or_extend()
s.plotear()