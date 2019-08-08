# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 11:50:30 2019

@author: Marcos
"""

import os
from utils import load_img, contenidos

target_dir = r'D:\Facu\Tesis\Códigos\Tesis\byc\sintetizados\as_loaded'

c  = contenidos(r'D:\Facu\Tesis\Códigos\Tesis\byc\sintetizados\dnn\train_pad')
c.extend(contenidos(r'D:\Facu\Tesis\Códigos\Tesis\byc\sintetizados\dnn\originales_pad'))
#%%
tipo = 'train', 'train', 'ori', 'ori'
for path, t in zip(c, tipo):
    i = load_img(path, target_size=(300, 200), color_mode='grayscale')
    nombre = os.path.join(target_dir, 
                          '_'.join((t, (os.path.basename(path)))))
    i.save(nombre)
#    print(nombre)
    
#%%

oris_act = contenidos(r'D:\Facu\Tesis\Códigos\Tesis\byc\modelos\byc_peque_pad')
oris_bent = contenidos(r'D:\Facu\Tesis\Códigos\Tesis\byc\nuevos\originales\sonos\pad\benteveo')
oris_chin = contenidos(r'D:\Facu\Tesis\Códigos\Tesis\byc\nuevos\originales\sonos\pad\chingolo')

oris_act = [f for f in oris_act if 'XC' in f]

archivos = []
for o in oris_act:
    base = o.split('_')[-1]
    if 'bent' in o:
        a = filter(lambda i: base in i, oris_bent)
    else:
        a = filter(lambda i: base in i, oris_chin)
    archivos.extend(a)
    
for f in archivos:
    i = load_img(f, target_size=(300, 200), color_mode='grayscale')
    nombre = os.path.join(target_dir, os.path.basename(f))
    i.save(nombre)