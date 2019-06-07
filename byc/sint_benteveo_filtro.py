# -*- coding: utf-8 -*-
"""
Created on march 2017

@author: Gabo Mindlin

Integrator with rk4, and tube with delays

it creates wav


"""

import os

path_sono = path_audio = 'filtro'

import numpy as np
from numpy.random import normal

from scipy.io.wavfile import write, read
from scipy import signal

import matplotlib.pyplot as plt
from utils import new_name

creo_nombre = lambda path, base, formato: new_name(os.path.join(path, base + formato))

# --------------------
# Parámetros generales
# --------------------
global estimulo1
global destimulodt1
i=0

for ch in [250, 350, 450]:
    for lg in [20, 35, 50]:
                
        gamma = 24000
        uoch, uolb, uolg, rb, rdis = (ch/2.0)*1e8, 0.1 , 1/lg, 1e7, 1e4   # 24*10000 , y con 350/3.0, la frec de la oec en 4000 Hz
        fsamp, L = 882000.0, 0.045
        dt = 1/fsamp
        tiempo_total = 0.86
        N=int((L/(350*dt))//1)
        
        cant_puntos = np.int(tiempo_total/(dt))
        
        
        # Function definitions
        # --------------------
        def ecuaciones(v, kappa, b):
            x,y,i1,i2,i3 = v
            dv = np.array((
                y,
                -gamma * gamma * kappa * (x + 0.1 * x * x * x) - gamma * (x * x + 0.3 * x * x * x) * y + b * gamma * y,
                i2,
                -uolg*uoch*i1-(rdis*uolb+rdis*uolg)*i2+(uolg*uoch-rdis*rb*uolg*uolb)*i3+uolg*destimulodt+rdis*uolg*uolb*estimulo,
                -(uolb/uolg) * i2 - rb * uolb * i3 + uolb * estimulo))
            return dv
        
        
        def rk4(dv, v, dt, kappa, b):
            dt2=dt/2.0
            dt6=dt/6.0
            k1 = dv(v, kappa, b)
            k2 = dv(v + dt2 * k1, kappa, b)
            k3 = dv(v + dt2 * k2, kappa, b)
            k4 = dv(v + dt * k3, kappa, b)
            v += dt6 * ( 2*(k2+k3) + k1 + k4)
            return v
        
        def forma_amps(inicio, fin):
            if inicio and fin:
                return lambda t: np.sin(np.pi * t)
            if inicio and not fin:
                return lambda t: np.sin(np.pi * t / 2)
            if not inicio and fin:
                return lambda t: np.sin((np.pi * (t+1) / 2))
            else: 
                return lambda t: 1                               

        def rectas(ti, tf, wi, wf, f, freqs, beta, amps, 
                    param=1, d=0, inicio=True, fin=True):
            
            if param not in (1,2):
                raise ValueError("parametrizacion debe ser 1 o 2")
                
            i=np.int(ti/dt)
            j=np.int(tf/dt)
            dj = j-i
            k = np.arange(dj) / dj # =  t - ti / (ti - tf)
            amps[i:j] = f * forma_amps(inicio, fin)(k)
        
            if param == 2:
                l = int(d/dt)
                i -= l  
                k = np.arange(-l, dj) / dj
                
            freqs[i:j] = wi + (wf-wi) * k
            beta[i:j] = .5
        

        #reinicio valores
        frecuencias=np.zeros(cant_puntos)
        beta = np.full(cant_puntos, -.10)
        amplitudes = np.zeros(cant_puntos)
        
        v = np.array([0.01, 0.001, 0.001, 0.0001, 0.0001])
    
        # -----------------------------------
        # Genero los parámetros de los cantos
        # ----------------------------------- 
    
        rectas(0.01, tiempo_total-0.01, 200, 5000,
               f=1, freqs=frecuencias, beta=beta, amps=amplitudes,
               inicio=False, fin=False)
        
        nombre_base = 'uoch={:.2e}_uolg={:.2f}'.format(uoch, uolg)
    
        #    #ploteo los parámetros
        #    tiempo = np.linspace(0, tiempo_total, cant_puntos)
        #    fig1, axs= plt.subplots(3,1, sharex=True)
        #    axs[0].plot(tiempo[::10],frecuencias[::10], '.')
        #    axs[1].plot(tiempo[::10],amplitudes[::10], '.')
        #    axs[2].plot(tiempo[::10],beta[::10], '.')
        
        # -------
        # Integro
        # -------
        v4 = []
        
        fil1 = np.zeros(N)
        back1 = np.zeros(N)
        
        print('integrando...')
        
        kappa_todos = (6.56867694e-08 * frecuencias*frecuencias + 4.23116382e-05 * frecuencias + 2.67280260e-02)
        b_todos = beta * normal(1, .1, cant_puntos)
        
        for kappa, b in zip(kappa_todos, b_todos):
            
            estimulo = fil1[-1]
            destimulodt = (fil1[-1] - fil1[-2]) / dt
            
            #integro
            rk4(ecuaciones,v,dt, kappa, b)
            
            #actualizo valores
            fil1[0]  = v[1] + back1[-1]
            back1[0] = -0.01 * fil1[-1]
            fil1[1:] = fil1[:-1] #desplazo todo 1 hacia el final
            back1[1:] = back1[:-1] #desplazo todo 1 hacia el final
              
            v4.append(v[4])
        
        
        sonido = np.array(v4) * amplitudes
    #    sonido *= 1000
    #    sonido += 10 * normal(0, .007, len(sonido))
        
        f, t, Sxx = signal.spectrogram(sonido, fsamp, window=('gaussian',20*128),
                                       nperseg=10*1024, noverlap=18*512, scaling='spectrum')
        
        fig, ax = plt.subplots()
        ax.pcolormesh(t,f,np.log10(Sxx),rasterized=True,
                      cmap=plt.get_cmap('Greys'))
        ax.set_ylim(10,8000)
    #    ax.axis('off')
        ax.grid()
        fig.subplots_adjust(bottom = 0, top = 1, left = .08, right = 1) #para que no tenga bordes blancos
        
        
        nombre = creo_nombre(path_sono, nombre_base, '.jpg')
        fig.savefig(nombre, dpi=100)
        plt.close()
        
        scaled = (sonido/np.max(np.abs(sonido))).astype(np.float32)
        nombre = creo_nombre(path_audio, nombre_base, '.wav')
        write(nombre, int(fsamp/20), scaled[::20])
            
        i +=1
        print('listo {} de {}(?)!'.format(i, 9))
print('\a') #sonido al final de la integración
            
                
#%%

lista = [os.path.join(path_sono,f) for f in os.listdir(path_sono) if f.endswith('wav')]

frecs = frecuencias[::20]
donde = np.where(frecs>0)[::50]

fig, axarr  = plt.subplots(3,3, sharex=True, sharey=True)

for f, ax in zip(lista, axarr.flatten()):
    _, signal = read(f)
    ax.plot(frecs[donde],signal[donde])
    ax.set_title(f)

axarr[-1,1].set_xlabel('frecuencia [Hz]')