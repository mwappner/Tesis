# -*- coding: utf-8 -*-
"""
Miro si hay cantos repetidos
"""

import os

def VecesQueAparece(completa, template):
    seguir = True
    apariciones = 0
    while seguir:
        try:
            out = completa.index(template) #encuentra el template
            completa = completa[out+1:] #recorta hasta donde encontró
            apariciones += 1 #cuenta una aparición
        except: #usa que .index() tira error si no encuentra nada
            seguir = False
                
    return apariciones
    
#%% Primero chequeo nombres

#directorio con grabaciones:
principal = '/home/marcos/Documents/Grabaciones/Finches'

#nombres de los archivos 
nombres = []
completos = []
for root,_,files in os.walk(principal):
    for file in files:
        nombres.append(file)
        completos.append(os.path.join(root,file))

repeticiones = []        
for nombre in nombres:
    repeticiones.append(VecesQueAparece(nombres,nombre)-1)

print('Cantidad de cosas con repeticiones (teniendo en cuenta que si a=b está contado, tabmién está contado b=a):',sum(repeticiones))
print('Máxima cantidad de repeticiones:',max(repeticiones))

#%% Elimino los archivos repetidos y guardo los índices

#como nada se repite mś de una vez, puedo recorrer la lista en orden y eliminar
#usando index()
lugares = [] #Contiene el par de indices repetidos
for ind,nombre in enumerate(nombres):
    try:
        donde = nombres[ind+1:].index(nombre)
        lugares.append([ind,donde+ind+1])
    except:
        continue

for par in lugares:
    print('El par repetido es:')
    print(completos[par[0]])
    print(completos[par[1]],'\n')
    
#### RESULTA QUE LAS DE SANTI SON REPETIDAS: las borré ####
    
#%% Chequeo si el canto en sí se repite de dor formas

#1. De forma exacta: 
from scipy.io import wavfile

cantos = []
duraciones = []
for file in completos:
    _,data = wavfile.read(file)
    cantos.append(data)
    duraciones.append(len(data))

repes = []
for d in duraciones:
    try:
        donde = duraciones[ind+1:].index(d)
        repes.append([ind,donde+ind+1])
    except:
        continue

if not repes: print('No tengo repetidas exactas.')
else: print('Estas son repetidas.') #Hacer algo al respecto...


#2. De forma aproximada (usando correlación):

######## HACER ##########

