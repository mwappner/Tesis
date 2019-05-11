# -*- coding: utf-8 -*-
"""
Created on march 2017

@author: Gabo Mindlin

Integrator with rk4, and tube with delays

it creates wav


"""

import os

cant_sintesis = 1 #cuantos cantos voy a sintetizar
nombre_base = 'benteveo' #nombre de los sonogramas
path_sono = os.path.join('nuevos', 'sonogramas')
path_audio = os.path.join('nuevos', 'audios')

import numpy as np
from numpy.random import normal

from scipy.io.wavfile import write
from scipy import signal

import matplotlib.pyplot as plt
from utils import new_name

creo_nombre = lambda path, base, formato: new_name(os.path.join(path, base + formato))

# --------------------
# Parámetros generales
# --------------------
global estimulo1
global destimulodt1

gamma = 24000
uoch, uolb, uolg, rb, rdis = (350/2.0)*100000000, 0.1 , 1/35., 10000000, 10000  # 24*10000 , y con 350/3.0, la frec de la oec en 4000 Hz
fsamp, L = 882000.0, 0.045
dt = 1/fsamp
tiempo_total = 1.2
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
            
def expo(ti, tf, wi, wf, f, freqs, beta, amps, tau=3, inicio=True, fin=True):
    
    i=np.int(ti/dt)
    j=np.int(tf/dt)
    k = np.arange(j-i) / (j-i) # =  t - ti / (tf - ti)

    freqs[i:j] = wf + (wi-wf) * np.exp(-tau * k)
    beta[i:j] = .5
    amps[i:j] = f * forma_amps(inicio, fin)(k)
#    amps[i:j] = f

def rectas(ti, tf, wi, wf, f, freqs, beta, amps, inicio=True, fin=True):
    
    i=np.int(ti/dt)
    j=np.int(tf/dt)
    k = np.arange(j-i) / (j-i) # =  t - ti / (ti - tf)

    freqs[i:j] = wi + (wf-wi) * k
    beta[i:j] = .5
    amps[i:j] = f * forma_amps(inicio, fin)(k)

def senito(ti, tf, media, amplitud, alphai, alphaf,
           f, freqs, beta, amps, param=1, d=0, inicio=True, fin=True):
    '''Param=1 corresponde a la aprametrización usual con (ti, tf, media, amplitud
    alphai, alphaf), mientras que param=2 corresponde a una parametrización que 
    deriva de lo anterior y que comienza en t<ti, por lo que pide d para armar un
    ti_nuevo = ti-d.'''
    
    if param not in (1,2):
        raise ValueError("parametrizacion debe ser 1 o 2")

    i=np.int(ti/dt)
    j=np.int(tf/dt)
    dj = j-i
#    k = np.arange(j-i)/j-i
    k = np.arange(0,1,1/dj)
    amps[i:j]= f * forma_amps(inicio, fin)(k)
    
    if param==1:
        freqs[i:j] = media + amplitud * np.sin(alphai + (alphaf - alphai) * k)
        beta[i:j] = .5
    else:
        l = int(d/dt)
        i2= i-l   
#        k = np.arange(-l/dj,1,1/dj)
#        k = np.linspace(i2, j, j-i2)
        k = np.arange(i2, j)
        
        phi = (alphai * j - alphaf * i) / dj
        omega = (alphaf - alphai) / dj
        freqs[i2:j] = media + amplitud * np.sin(phi + omega * k )
        beta[i2:j] = .5
#    new_k = 5* k / (j-i) # = k/tau
#    amps[i:j] = f * new_k * np.exp(-new_k) * normal(1, .1) * (1 + .4 * np.sin(2*np.pi * k / 6820))
#    amps[i:j] = f


#%%
for i in range(cant_sintesis):
    
    #reinicio valores
    frecuencias=np.zeros(cant_puntos)
    beta = np.full(cant_puntos, -.10)
    amplitudes = np.zeros(cant_puntos)
    
    v = np.array([0.01, 0.001, 0.001, 0.0001, 0.0001])

# -----------------------------------
# Genero los parámetros de los cantos
# ----------------------------------- 
    f = 0.35
    senito(ti=0.184, tf=0.33, media=1750, amplitud=70, alphai=2.4, alphaf=0.7,
           f=f, freqs=frecuencias, beta=beta, amps=amplitudes, param=2, d=0.05)

    senito(ti=0.59, tf=0.64, media=-870, amplitud=2960, alphai=2.35, alphaf=1.34,
           f=f, freqs=frecuencias, beta=beta, amps=amplitudes, param=2, d= 0.05, fin=False)
    expo(ti=0.64, tf=0.69, wi=2010, wf=160, tau=0.68,
         f=f, freqs=frecuencias, beta=beta, amps=amplitudes, inicio=False)
    
    senito(ti=0.737, tf=1.054, media=1290, amplitud=570, alphai=9.7, alphaf=6,
           f=f, freqs=frecuencias, beta=beta, amps=amplitudes, param=2, d = 0.03)

#    senito(0.166,0.32+0.05,1310*0.5,200*0.5,0,np.pi,0.7*1.1,frecuencias,beta,amplitudes)
#    senito(0.58,0.7,1305*0.5,600*0.5,-np.pi/4.0,3*np.pi/2.0,0.7*1,frecuencias,beta,amplitudes)
#    senito(0.74+0.05,1.06,1301,200,0,np.pi+np.pi/4.0,0.7*1,frecuencias,beta,amplitudes)
#
    tiempo = np.linspace(0, tiempo_total, cant_puntos)
    fig1, axs= plt.subplots(3,1, sharex=True)
    axs[0].plot(tiempo[::10],frecuencias[::10], '.')
    axs[1].plot(tiempo[::10],amplitudes[::10], '.')
    axs[2].plot(tiempo[::10],beta[::10], '.')

#%%
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
    sonido *= 1000
    sonido += 20 * normal(0, .007, len(sonido))
    
    f, t, Sxx = signal.spectrogram(sonido, fsamp, window=('gaussian',20*128),
                                   nperseg=10*1024, noverlap=18*512, scaling='spectrum')
    
    fig, ax = plt.subplots()
    ax.pcolormesh(t,f,np.log10(Sxx),rasterized=True,
                  cmap=plt.get_cmap('Greys'))
    ax.set_ylim(10,15000)
    ax.axis('off')
    fig.subplots_adjust(bottom = 0, top = 1, left = 0, right = 1) #para que no tenga bordes blancos
    
    nombre = creo_nombre(path_sono, nombre_base, '.jpg')
    fig.savefig(nombre, dpi=100)
    plt.close()
    
    scaled = (sonido/np.max(np.abs(sonido))).astype(np.float32)
    nombre = creo_nombre(path_audio, nombre_base, '.wav')
    write(nombre, int(fsamp), scaled)
    
    
    print('listo {} de {}!'.format(i+1, cant_sintesis))
    print('\a') #sonido al final de la integración
    
        
