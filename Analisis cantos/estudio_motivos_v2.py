#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 15:27:31 2018

@author: marcos
"""

import silabas as si
import os
import numpy as np
import matplotlib.pyplot as plt
import itertools
from skimage.filters import threshold_otsu

#Consigo todos los archivos
home = os.getcwd()
ubicacion = os.path.join(home, 'Motivos', 'Secuencias')
ubi_nuevos = os.path.join(ubicacion,'Nuevos')
ubi_viejos = os.path.join(ubicacion,'Server')
archivos = [os.path.join(ubi_nuevos, f) for f in os.listdir(ubi_nuevos)]
archivos.extend([os.path.join(ubi_viejos, f) for f in os.listdir(ubi_viejos)])
archivos.sort() #archivos ordenados alfabeticamente!!

#%% Leo los motivos 

#cargo todas las sílabas
motivos = []
secuencias = []
for file in archivos:
    motivos.extend(si.procesar_archivo(file))

#arreglo formato, extraigo duraciones y notas de potenciales highnotes:

for gesto in motivos:
    gesto.hago_pendiente()

#después de haber agregado 'bajada' a la lista
duraciones = {c:[] for c in si.categorias}
for gesto in motivos:
    duraciones[gesto.categoria].append(gesto.duracion)
   
#%% Grafico histograma de duraciones

for k, cat in enumerate(si.categorias):
    ax = plt.subplot(3,3,k+1)
    ax.set_title('{} ({} totales)'.format(cat,len(duraciones[cat])))
    ax.hist(duraciones[cat],range = [0,0.12])
    ax.set_xlim([0,0.12])
    ax.set_ylabel('Ocurrencia')
    ax.set_xlabel('Duración [s]')
plt.tight_layout()

### SE VE SEPARACIÓN de dos tipos de silencio

#%% Separo en dos tipos de silencio:

threshold = threshold_otsu(np.array(duraciones['silencio']))

#Convierto los silencios cortos de categoría.
for gesto in motivos:
    gesto.separo_categoria(threshold, si.conversor['s'], 'sil_corto')
   
#Redefino duraciones:
duraciones = {c:[] for c in si.categorias}
for gesto in motivos:
    duraciones[gesto.categoria].append(gesto.duracion)
    
#%% Vuelvo a plotear
    
salteado = False
for k, cat in enumerate(si.categorias):
    if cat=='seno':
        salteado = True
        continue
    if salteado:
        k -= 1
    ax = plt.subplot(3,3,k+1)
    ax.set_title('{} ({} totales)'.format(cat,len(duraciones[cat])))
    ax.hist(duraciones[cat],range = [0,0.12])
    ax.set_xlim([0,0.12])
plt.tight_layout()


#%% Pendientes
#TODO:
    #continuidad en frecuencia (implementado en versión anterior)
    #parametro de las exp.
    #parametros de cosenos
    #percu/varias

#%% Miro secuencias

#Una lista de 4 elementos donde cada una contiene todas las combinaciones 
#posibles de 2, 3, 4 y 5 letras respectivas.

categorias = si.categorias
abecedario = [chr(c) for c in range(ord('a'),ord('z'))]
codigo = {cat:abc for cat, abc in zip(categorias, abecedario)}
inv_codigo = {abc:cat for cat, abc in codigo.items()}

secuencias = si.armar_secuencias(motivos, codigo)

o = []
for orden in range(4):
    o.append([''.join(l) for l in itertools.product(sorted(codigo.values()), repeat=orden+2)])


#%% Hallo secuencias

#paraacelerarlo podría usar que las secuencias de longitud n contienen 
#las de longitud n-1 y que por ende podría reducir la búsqueda después 
#de la primera pasada guardando los índices de dódne encontré patrones.

veces = [] #contiene la cantidad de veces que aparece cada transición
for orden in o:
    esta_vez = []
    for sec in orden:
        total = 0
        for canto in secuencias:
            total += si.VecesQueAparece(canto,sec)
        if total == 0: total = np.nan
        esta_vez.append(total)
    veces.append(esta_vez)

# los arrays oi tienen la frecuencia de ocurrencia de cada transición

#cada columna tiene la cantidad de veces que se pasa de un estado dado
#la fila indica a qué estado

#ejemplos (orden 1): [0,0] = a-->a; [0,3] = a-->d; [3,5] = d-->f
o1 = np.reshape(veces[0],[len(codigo),len(codigo)]).T

#ejemplos (orden 2): [0,0] = aa-->a; [0,3] = aa-->d; [3,5] = ad-->f
o2 = np.reshape(veces[1],[len(codigo)**2,len(codigo)]).T

#ejemplos (orden 3): [0,0] = aa-->aa; [0,3] = aa-->ad; [3,5] = ad-->af
o3 = np.reshape(veces[2],[len(codigo)**2,len(codigo)**2]).T


#%% Grafico las matrices

plt.matshow(o1)
plt.colorbar()
plt.xticks(range(len(codigo)),sorted(codigo.values()))
plt.yticks(range(len(codigo)),sorted(codigo.values()))

plt.matshow(o2)
plt.xticks(range(len(codigo)**2),[c[-1] for c in o[0]])
plt.yticks(range(len(codigo)),sorted(codigo.values()))
plt.colorbar()

plt.matshow(o3)
plt.colorbar()
plt.xticks(range(len(codigo)**2),[c[-1] for c in o[0]])
plt.yticks(range(len(codigo)**2),[c[-1] for c in o[0]])

#%% Hago las cosas bien normalizadas

# Normalizo por columna: probabilidad de transición A ESE ESTADO.
# Como la cantidad de eventos en cada columna es igual a la cantidad de 
# que aparece el gesto de partida, normalizo por eso

#frecuencia de aparición de cada gesto
f0 = [len(duraciones[cat]) for cat in categorias]

#usando broadcasting sobre matrices cuadradas, toma el array a brocastear
#como fila y lo repite varias veces (que es lo que necesito). Por ejemplo:

# |1 2| * |1,2| = |1*1 2*2|
# |3 4|           |3*1 4*2|
n1 = o1 / f0

#%% Miro probabilidad del primer gesto y frecuencia por momento


#Calcula las cuentas de cada categoría para un histograma categórico
def Cuentas(datos,categs,diccionario):
#toma una lista conteniendo los gestos codificados
    out = np.zeros(len(categs))
    for dato in datos:
        ind = categs.index(diccionario[dato])
        out[ind] += 1
        
    return out

def ElHisto(lascuentas):    
    plt.bar(range(len(lascuentas)),lascuentas,align='center')
    plt.xticks(range(len(lascuentas)),categorias,rotation=45)
        
#miro el primer gesto en cada uno
primeros = []
for sec in secuencias:
    primeros.append(sec[0])

cant_primeros = Cuentas(primeros,categorias,inv_codigo) 
ElHisto(cant_primeros)
#plt.bar(range(len(categorias)),cant_primeros,align='center')
#plt.xticks(range(len(categorias)),categorias,rotation=45)

#frecuenca total de cada gesto:
cant_totales = np.zeros(len(categorias))
for sec in secuencias:
    cant_totales += Cuentas(sec,categorias,inv_codigo)

ElHisto(cant_totales)

cant_inicio = np.zeros(len(categorias))
cant_medio= np.zeros(len(categorias))
cant_final= np.zeros(len(categorias))

for sec in secuencias:
    l = len(sec)
    cant_inicio += Cuentas(sec[:l//3],categorias,inv_codigo)
    cant_medio += Cuentas(sec[l//3:l//3*2],categorias,inv_codigo)
    cant_final += Cuentas(sec[2*l//3:],categorias,inv_codigo)
   

plt.subplot(411)
ElHisto(cant_inicio/sum(cant_inicio))
plt.subplot(412)
ElHisto(cant_medio/sum(cant_medio))
plt.subplot(413)
ElHisto(cant_final/sum(cant_final))
plt.subplot(414)
ElHisto(cant_primeros)

ultimos = []
for sec in secuencias:
    ultimos.append(sec[-1])
    
cant_ultimos = Cuentas(ultimos,categorias,inv_codigo) 
ElHisto(cant_ultimos)