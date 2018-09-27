# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 12:06:52 2018

@author: marcos
"""

import os
import numpy as np
import matplotlib.pyplot as plt


ubicacion = 'Motivos/Secuencias'
archivos = os.listdir(ubicacion)
archivos.sort() #archivos ordenados alfabeticamente!!
#%% Leo los motivos 

motivos = []
for file in archivos:
    este = open(os.path.join(ubicacion,file),'r')
    temp = este.readlines()
    este.close()
    temp = [linea[:-1] for linea in temp] #le saco el \n del final
    motivos.append(temp)
    
#Cuando hay, saco la línea vacía al principio del archivo.
#Elimino las líneas con 'no'
for i,mot in enumerate(motivos):
    #borro líneas vacías
    #borro las líneas con 'no'
    for j,linea in enumerate(mot):

        if linea[-2:] == 'no' or not linea:
            del mot[j]

#%% Guardo todo junto y hextraigo duraciones

#guardo TODO en una única lista, separo cada renglón en sus elementos
todo = []
for mot in motivos:
    for linea in mot:
        
        #separo en los parámetros
        linea = linea.split('-')
        
        #arreglo errores:
        if len(linea[-1].split('.')) == 3 : #si la dur. es tipo 0.0.03, arreglo
            linea[-1] = linea[-1][2:]
        if linea[0] == 'd': linea[0] = 's' #'d' no es categoría, pongo 's'
        if float(linea[-1]) == 0 : linea[-1] = np.nan #salteo cosas con duración 0
        if linea[0] == 'C': linea[0] = 'c'
        if linea[0] == 'h' and len(linea) == 4: linea[0] = 'b'
            
        #convierto a floats los números
        linea[1:] = [float(item) for item in linea[1:]] 
        todo.append(linea)


categorias = ['ruido','constante','silencio','exp','high','percu','seno','varias','subida','bajada']
letras = ['r', 'k','s','e','h','p','c','v','b']

#duracio de cada gesto
duraciones = {categ:[] for categ in categorias}
#nota de las cosas que pueden ser high notes
notas = {categ:[] for categ in ['high','constante','bajada','subida']}


#b lo redefino como b [ordenada] [pendiente] [duracion]
for linea in todo:
    if linea[0] == 'b':
        pend = (linea[2]-linea[1])/linea[-1]
        if pend>0: linea[0] = categorias[-2] #subida
        else: linea[0] = categorias[-1] #bajada
            
        linea[2] = pend
        
        
    else: linea[0] = categorias[letras.index(linea[0])]
    
    #lleno los arrays de datos
    duraciones[linea[0]].append(linea[-1])
    if linea[0] in notas.keys(): notas[linea[0]].append(linea[1])

#%% Histogramas de duraciones

k = 0
for cat in categorias:
    if cat == 'seno':
        print('Cantidad de senos: {}'.format(len(duraciones[cat])))        
        continue
    k += 1
    ax = plt.subplot(3,3,k)
    ax.set_title('{} ({} totales)'.format(cat,len(duraciones[cat])))
    ax.hist(duraciones[cat],range = [0,0.12])
    ax.set_xlim([0,0.12])
    
#%% Histograma de notas
    
fig,ax = plt.subplots(3,1,sharex='all')
cat = list(notas.keys()) #contenido de notas
del cat[cat.index('high')] #para graficar todas contra high

for k,c in enumerate(cat):
    ax[k].hist(notas[c] , alpha = 0.3 ,label = 'otro', normed = 1, bins=20)
    ax[k].hist(notas['high'] , alpha = 0.3 ,label = '{} ({})'.format('high',len(notas['high'])), normed = 1, range=[0,0.1])
    ax[k].set_title('{} ({} totales)'.format(c,len(notas[c])))
    plt.legend()

plt.legend()


#%% PENDIENTES:

#parametro de las exp.
#continuidad en frecuencia
#parametros de cosenos
#percu/varias

#%% Continuidad de frecuencias

#sólo miro continuidad de k,b y h.
#Debería incluirse c y eventualmente p/v
utiles = ['constante','high','subida','bajada','exp']

#reestructuro los datos de una forma útil: [cat],[f_in],[f_fin]
frecuencias = []
sec = []
m = 0
k = 0
for linea in todo:
    #trato por separado subidas y bajadas de lo otro
    if linea[0] in utiles[:2]: #cte. o high
        sec.append([linea[1],linea[1]])
    elif linea[0] in utiles[2:]: #bajada, subida o exponencial
        f_fin = linea[1] + linea[2] * linea[3]
        sec.append([linea[1],f_fin])
    else: #no es de las útiles
        sec.append([np.nan,np.nan])
    k += 1
    #separo por canto para no tomar continuidades incorrectas
    if k == len(motivos[m]):
        #guardo al diferencia de fecuencias inicial y final:
        frecs = np.array(sec)
        frecs = np.abs(frecs[:-1,1]-frecs[1:,0]) 
        frecuencias.append(frecs) #uno toda la secuencia en un único string.
        sec = []
        m += 1
        k = 0

#vuelvo a concatenar todo:
diferencias = np.concatenate(frecuencias)
diferencias = diferencias[~np.isnan(diferencias)] #me deshago de los nan

#hago un histograma para elegiur el threshold de continuidad
#plt.hist(diferencias,bins=1000)

threshold = 0.005

#cantidad de transiciones continuas en función del threshols:
threshs = np.arange(0,0.1,0.001)
cant_cont = np.array([sum(diferencias<thresh) for thresh in threshs])

plt.plot(threshs,cant_cont,'o-')
plt.plot(np.ones(2)*threshold,[0,500])

prob_cont = sum(diferencias<threshold) / len(diferencias)
print('Probabilidad de que una transición sea contínua: {:.3f}'.format(prob_cont))


#%% Creo secuencias

#Recupero las secuencias de cada canto (sin parametros)
abecedario = [chr(c) for c in range(ord('a'),ord('z'))]
codigo = {categorias[k]:abecedario[k] for k in range(len(categorias))}
inv_codigo = {abecedario[k]:categorias[k] for k in range(len(categorias))}

secuencias = []
sec = []
m = 0
k = 0
for linea in todo:
    sec.append(codigo[linea[0]])
    k += 1
    if k == len(motivos[m]):
        secuencias.append(''.join(sec)) #uno toda la secuencia en un único string.
        sec = []
        m += 1
        k = 0

#chequeo que estén todos. Si printea 0, está bien
print(sum([len(sec)-len(motivos[k]) for k,sec in enumerate(secuencias)]))

#Creo las secuencias para contrastar
import itertools

#transiciones de orden 1 a 4
o = []
#creo todas las combinaciones posibles de las letras de los códigos
#para eso uso product() y le pido que haga secuencias de longitud orden+2

for orden in range(4):
    o.append([''.join(l) for l in itertools.product(sorted(codigo.values()), repeat=orden+2)])
#o1 = [''.join(l) for l in itertools.product(codigo.values(), repeat=2)]
#o2 = [''.join(l) for l in itertools.product(codigo.values(), repeat=3)]
#o3 = [''.join(l) for l in itertools.product(codigo.values(), repeat=4)]
#o4 = [''.join(l) for l in itertools.product(codigo.values(), repeat=5)]



#%% Hallo secuencias

#Cuento las veces que aparece un string en otro string o elemento en lista
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

veces = [] #contiene la cantidad de veces que aparece cada transición
for orden in o:
    esta_vez = []
    for sec in orden:
        total = 0
        for canto in secuencias:
            total += VecesQueAparece(canto,sec)
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