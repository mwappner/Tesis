#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 14:48:51 2018

@author: marcos
"""
import os
#%%
categorias = ('ruido','constante','silencio','exp','percu','seno','varias','subida','bajada')
letras = ('r', 'k','s','e','p','c','v','b')
leyenda_datos = ((''), ('frec'), (''), ('frec_i', 'frec_f'),
                 ('frec_i', 'frec_f', 'cant'), (''), 
                 ('frec_i', 'frec_f', 'cant'), ('ordenada', 'pendiente'),
                 ('ordenada', 'pendiente'))
conversor = dict(zip(letras, categorias))

abecedario = [chr(c) for c in range(ord('a'),ord('z'))]
codigo = {categorias[k]:abecedario[k] for k in range(len(categorias))}
inv_codigo = {abecedario[k]:categorias[k] for k in range(len(categorias))}

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
    
    #chequeo si es de los nuevos o los viejos
    direc, _ = os.path.split(file)
    es_nuevo = os.path.basename(direc) == 'Nuevo'
       
    #guardo objetos sílaba en la lista que devuelvo
    motivo = [Gesto(linea, file, nuevo=es_nuevo) for linea in temp]
    
    #guardo la secuencia de sílabas de este archivo
    secuencia = [codigo[g.categoria] for g in motivo]
    
    return motivo, ''.join(secuencia)

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
    '''Clase que guarda los datos de una dada sílaba.'''
    
    def __init__(self, string, parent, nuevo=False):
        
        #separo el string en los datos, guardo el string y corrijo typos
        datos = string.split('-')
        self.original = string
        datos = corrijo_typos(datos)

        #define duración y si es nevo o no
        self.duracion = float(datos.pop())
        self.nuevo = nuevo
        
        #define dónde está en el canto. sólo apra datos neuvos.
        if nuevo:
            self.ubicacion = float(datos.pop())
        else:
            self.ubicacion = None
        
        #cordefinir categoría, redefino las mayúsculas a minúsculas
        self.categoria = conversor.get(datos.pop(0).lower(), None)
        
        #definir de qué motivo proviene
        self.archivo = parent
        
        #el resto de los datos
        self.data = [float(d) for d in datos]
        
    def hago_pendiente(self):
        '''Redefino los datos de los barridos para que sean (ordenada, pendiente)
        y los guardo así. Por default, todos los barridos son subidas. Cuando
        corresponda, defino la categoría como bajada.'''
        
        if self.categoria == conversor['b']:
            
            pendiente = (self.data[1] - self.data[0]) / self.duracion
            
            if pendiente<0: 
                self.categoria = categorias[-1] #bajada
             
            self.data[1] = pendiente