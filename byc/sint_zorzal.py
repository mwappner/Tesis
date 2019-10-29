# -*- coding: utf-8 -*-
"""
Created on march 2017

@author: Gabo Mindlin

Integrator with rk4, and tube with delays

it creates wav


"""
import os

cant_sintesis = 1 #cuantos cantos voy a sintetizar
nombre_base = 'zorzal' #nombre de los sonogramas
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

tiempo_total=1.94

gamma=24000
escala = 1.6 #factor de escala para pasar de chingolo a zorzal
uoch, uolb, uolg, rb, rdis = (350/5.0)*1e8, 1e-4, 1/20., 5e6, 2.4e5 # 2.4e5 , y con 350/3.0, la frec de la oec en 4000 Hz
fsamp, L=  882000.0, 0.025

#reescalo componentes: 
uoch *= escala**3
uolb *= escala
uolg *= escala
L *= escala

#calculo cantidades asociadas
dt = 1/fsamp
N=int((L/(350*dt))//1) #retardo del sonido (por la refelxión)

cant_puntos = np.int(tiempo_total/(dt))

# --------------------
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
            
def expo(ti, tf, wi, wf, f, freqs, beta, amps, tau=3,
         param=1, d=0, inicio=True, fin=True):
    
    i=np.int(ti/dt)
    j=np.int(tf/dt)
    dj = j-i
    k = np.arange(dj) / dj # =  t - ti / (tf - ti)
    amps[i:j] = f * forma_amps(inicio, fin)(k)
    
    if param == 2:
        l = int(d/dt)
        i -= l  
        k = np.arange(-l, dj) / dj

    freqs[i:j] = wf + (wi-wf) * np.exp(-tau * k)
    beta[i:j] = .5

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
    k = np.arange(0,1,1/dj)
    amps[i:j]= f * forma_amps(inicio, fin)(k)
    
    if param==1:
        freqs[i:j] = media + amplitud * np.sin(alphai + (alphaf - alphai) * k)
        beta[i:j] = .5
    else:
        phi = (alphai * j - alphaf * i) / dj
        omega = (alphaf - alphai) / dj
        
        l = int(d/dt)
        i -= l   
        k = np.arange(i, j)
        
        freqs[i:j] = media + amplitud * np.sin(phi + omega * k )
        beta[i:j] = .5


#%%
#reinicio valores
for _ in range(cant_sintesis):
    
    frecuencias=np.zeros(cant_puntos)
    amplitudes=np.zeros(cant_puntos)
    beta = np.full(cant_puntos, -1.0)
    
    v = np.array([0.01, 0.001, 0.001, 0.0001, 0.0001])
    
    # -----------------------------------
    # Genero los parámetros de los cantos
    # -----------------------------------
    
    ### XC351066 entre 5.85 y 7.79
    
    t1, medio1, t2 = 0.03, 0.13, 0.25
    w1, w2, w3, w4 = 2222, 2350, 1870, 1950
    rectas(t1, medio1, w1, w2, 1, frecuencias, beta, amplitudes, fin=False, param=2, d=0.02)
    rectas(medio1, t2, w3, w4, 1, frecuencias, beta, amplitudes, inicio=False)
    d = 0.36
    rectas(t1+0.02+d, medio1+d, w1, w2, 1, frecuencias, beta, amplitudes, fin=False, param=2, d=0.02)
    rectas(medio1+d, t2+d, w3, w4, 1, frecuencias, beta, amplitudes, inicio=False)
    
    medio2, medio3, medio4 = 0.82, 0.89, 1.03
    w1, w2 = 2550, 2681
    rectas(0.8, medio2, 2167, w1, 1, frecuencias, beta, amplitudes, fin=False, param=2, d=0.05)
    rectas(medio2, medio3, w1, w2, 1, frecuencias, beta, amplitudes, fin=False, inicio=False)
    rectas(medio3, 0.92, w2, 3190, 1, frecuencias, beta, amplitudes, inicio=False)
    expo(1.01, medio4, 2190, 2573, 1, frecuencias, beta, amplitudes, tau=0.8, fin=False, param=2, d=0.05)
    rectas(medio4, 1.12, 2394, 2340, 1, frecuencias, beta, amplitudes, inicio=False)
    
    t1, medio1, medio2, t2 = 1.38, 1.4, 1.47, 1.53
    w1, w2, w3, w4, w5, w6 = 1600, 2060, 1845, 1790, 2222, 1830
    expo(t1, medio1, w1, w2, 1, frecuencias, beta, amplitudes, tau=0.8, fin=False, param=2, d=0.05)
    rectas(medio1, medio2, w3, w4, 1, frecuencias, beta, amplitudes, inicio=False)
    rectas(medio2, t2, w5, w6, 1, frecuencias, beta, amplitudes)
    d = 0.3
    expo(t1+d, medio1+d, w1, w2, 1, frecuencias, beta, amplitudes, tau=0.8, fin=False, param=2, d=0.05)
    rectas(medio1+d, medio2+d, w3, w4, 1, frecuencias, beta, amplitudes, inicio=False)
    rectas(medio2+d, t2+d, w5, w6, 1, frecuencias, beta, amplitudes)

    
    tiempo = np.linspace(0, tiempo_total, cant_puntos)
    fig1, axs= plt.subplots(3,1, sharex=True)
    axs[0].plot(tiempo[::10],frecuencias[::10], '.')
    axs[1].plot(tiempo[::10],amplitudes[::10], '.')
    axs[2].plot(tiempo[::10],beta[::10], '.')
#%%
    # -------
    # Integro
    # -------
    v4=[]
    
    fil1=np.zeros(N)
    back1=np.zeros(N)
    
    print('integrando...')
    
    kappa_todos = 6.56867694e-08*frecuencias*frecuencias+4.23116382e-05*frecuencias+2.67280260e-02
    b_todos = beta * normal(1, .05)
    
    
    for kappa, b in zip(kappa_todos, b_todos):
        
        estimulo=fil1[N-1]
        destimulodt=(fil1[N-1]-fil1[N-2])/dt
        
        rk4(ecuaciones,v,dt, kappa, b)
        
        fil1[0]=v[1]+back1[N-1]
        back1[0]=-0.65*fil1[N-1]
        fil1[1:]=fil1[:-1]
        back1[1:]=back1[:-1]
    
        v4.append(v[4])
        # sonido[cont]=back[0]
        
    sonido = (np.array(v4) * amplitudes * normal(1,0.01))
    sonido *= 1000
    sonido += 5*normal(0, .007, cant_puntos)
    
    f, t, Sxx = signal.spectrogram(sonido,fsamp,window=('gaussian',20*128),
                                   nperseg=10*1024,noverlap=18*512,scaling='spectrum')
    fig, ax = plt.subplots()
    ax.pcolormesh(t,f,np.log10(Sxx),rasterized=True,
                  cmap=plt.get_cmap('Greys'))
    ax.set_ylim(10,8000)
    ax.axis('off')
    fig.subplots_adjust(bottom = 0, top = 1, left = 0, right = 1) #para que no tenga bordes blancos
    
    nombre = creo_nombre(path_sono, nombre_base, '.jpg')
    fig.savefig(nombre, dpi=100)
    plt.close()
    
    scaled = (sonido/np.max(np.abs(sonido))).astype(np.float32)
    nombre = creo_nombre(path_audio, nombre_base, '.wav')
    write(nombre, int(fsamp/20), scaled[::20])
    
    print('listo!')
    print('\a') #sonido al final de la integración
    

