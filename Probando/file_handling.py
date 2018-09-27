# -*- coding: utf-8 -*-
"""
File read and write
"""

import os
print(os.getcwd())
#%%
days_f = open('days.txt','r')
days_f.read()
days_f.readline() #no devuelve nada porque el comando anterior llevó el cursor de lectura al final del archivo
#si quiero que devulva algo, tengo que reabrirlo:

days_f = open('days.txt','r')
print(days_f.readline()) #primera línea
days_f.readlines() #el resto de las líneas

#%% escribir

title = 'Days of the week\n\n'

days_f = open('days.txt','r')
days = days_f.read() #contiene el conenido del archivo en un único string
nuevo = 'new_days.txt'
nd = open(nuevo,'w') 
#'w' no agrega sobre el conenido de un archivo, lo sobreescribe
#usar 'a' para append en un archivo existente

nd.write(title)
nd.write(days)
nd.close()
days_f.close()

