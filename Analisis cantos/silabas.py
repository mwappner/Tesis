#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 14:48:51 2018

@author: marcos
"""
import os
from orderedset import OrderedSet
#%% Parámetros generales

leyenda_datos = dict(
                 ruido=('',), 
                 constante=('frec',), 
                 silencio=('',), 
                 exp=('frec_i', 'frec_f',),
                 percu=('frec_i', 'frec_f', 'cant',), 
                 seno=('',), 
                 varias=('frec_i', 'frec_f', 'cant',), 
                 subida=('ordenada', 'pendiente',),
                 )

categorias = OrderedSet(leyenda_datos.keys())
letras = ('r', 'k','s','e','p','c','v','b')
conversor = dict(zip(letras, categorias))

#%% Funciones and stuff

def corrijo_typos(datos):
    '''Corrijo toda una serie de typos que encontré. Sujeto a agregar más.'''
        
    if len(datos[-1].split('.')) > 2 : #si la dur. es tipo 0.0.03, arreglo
        datos[-1] = datos[-1][2:]
        
    if datos[0] == 'd':
        datos[0] = 's' #'d' no es categoría, pongo 's'
        
    if float(datos[-1]) == 0 :
        datos[-1] = None #salteo cosas con duración 0

    #redefino highnotes como barridos o constantes, sewgún cantidad de parámetros
    if datos[0] == 'h':
        if len(datos) == 4:
            datos[0] = 'b'
        else:
            datos[0] = 'k'
    
    return datos

def quito_barraene(string):
    #Si la línea termina con '\n', se lo saco, si no, no hago nada
    if string.endswith('\n'):
        string = string[:-1]
    
    return string
    
def procesar_archivo(file):
    '''Carga un archivo y guarda todas las sílabas.'''
    
    with open(file, 'r') as este:
        temp = este.readlines()

    #borro el \n final
    temp = [quito_barraene(linea) for linea in temp]

    #borro líneas vacías
    #borro las líneas con 'no'
    temp = [linea for linea in temp if (not linea[-2:] == 'no' and linea)] 
    
    #chequeo si es de los nuevos o los viejos en función de en qué carpeta está
    es_nuevo = os.path.basename(os.path.split(file)[0]) == 'Nuevos'
       
    #guardo objetos sílaba en la lista que devuelvo
    motivo = [Gesto(linea, file, nuevo=es_nuevo) for linea in temp]
    
    return motivo

#Cuento las veces que aparece un string en otro string o elemento en lista
def VecesQueAparece(completa, template):
    seguir = True
    apariciones = 0
    while seguir:
        try:
            out = completa.index(template) #encuentra el template
            completa = completa[out+1:] #recorta hasta donde encontró
            apariciones += 1 #cuenta una aparición
        except ValueError: #usa que .index() tira error si no encuentra nada
            seguir = False
                
    return apariciones


def armar_secuencias(motivos, codigo):
    '''Armo las secuencias de cada canto de acuerdo al código'''
    
    nombre_actual = ''
    secuencias = []
    sec = ''
    for gesto in motivos:
        
        if nombre_actual == gesto.archivo:
            #si estoy en el mismo archivo, apendeo la categoria
            sec += codigo[gesto.categoria]
        else:
            #si no, guardo la secuencia e inicio una nueva
            secuencias.append(sec)
            sec = ''
            nombre_actual = gesto.archivo
            
    return secuencias

##No está andando
#notas = {categ:[] for categ in ['high','constante','bajada','subida']}
#def creo_notas(silaba, notas_so_far):
#    '''Agrego notas de los gestos potencialmente highnotes.'''
#    try:
#        if silaba.categoria in notas.keys():
#            try:
#                notas_so_far[silaba.categoria].append(silaba.data[0])
#            except:
#                    pass
#    except:
#        notas_so_far = notas
#    return notas_so_far


class Gesto:
    '''Clase que guarda los datos de una dada sílaba.
    
    Atributos por defecto:
        original : str
            el string orignial que codifica el gesto
        duracion : float
            duración del gesto
        ubicación : float
            timestamp de la ubicación en el sonograma
        nuevo : bool
            si es gesto de la primera o segunda serie de datos analizados
        categoria : string {'ruido','constante','silencio','exp', 'percu','seno',
                            'varias','subida', 'bajada', 'sil_corto', ...}
            categoría a la uqe pertenece el gesto. Pueden agregarse nuevas
            categorías durante el programa
        categ_orig : str {'r', 'k','s','e','p','c','v','b'}
            categoría original como se codifica en Gesto.original
        archivo : str (path)
            de qué archivo está tomado originalmente el gesto (path completo)
        data : list
            parámetros del gesto'''
    
    def __init__(self, string, parent, nuevo=False):
        
        #separo el string en los datos, guardo el string y corrijo typos
        datos = string.split('-')
        self.original = string
        datos = corrijo_typos(datos)

        #define duración y si es nevo o no
        self.duracion = float(datos.pop())
        self.nuevo = nuevo
        
        #define dónde está en el canto. sólo para datos neuvos.
        if nuevo:
            self.ubicacion = float(datos.pop())
        else:
            self.ubicacion = None
        
        #cordefinir categoría, redefino las mayúsculas a minúsculas
        self.cat_orig = datos.pop(0).lower()
        self.categoria = conversor.get(self.cat_orig, None)
        
        #definir de qué motivo proviene
        self.archivo = parent
        
        #el resto de los datos
        self.data = [float(d) for d in datos]
        
    def hago_pendiente(self):
        '''Redefino los datos de los barridos para que sean (ordenada, pendiente)
        y los guardo así. Por default, todos los barridos son subidas. Cuando
        corresponda, defino la categoría como bajada.'''
                
        if self.cat_orig == 'b':
            
            pendiente = (self.data[1] - self.data[0]) / self.duracion
            
            #por default son subidas, pongo 'bajda' cuando corresponde
            if pendiente<0:
                
                #agrego bajada a la lista de categorias
                nueva_cat = 'bajada'
                global categorias
                categorias.add(nueva_cat)
                
                #agrego los parámetros de bajada
                global leyenda_datos
                leyenda_datos[nueva_cat] = leyenda_datos[self.categoria]
        
                self.categoria = nueva_cat #bajada
             
            self.data[1] = pendiente
        
    def separo_categoria(self, threshold, categoria, nueva_cat):
        '''Separo los elementos de una categoría por duración: los mayores a
        threshold se quedan como estaban, los otros pasan a una nueva categoría'''
                
        if self.categoria == categoria:
            if self.duracion < threshold:
                self.categoria = nueva_cat
                
                #agrego la nueva categoría a la lista
                global categorias
                categorias.add(nueva_cat)
                
                #agrego los pàrámetros de datos correspondientes
                global leyenda_datos
                leyenda_datos[nueva_cat] = leyenda_datos[categoria]
            