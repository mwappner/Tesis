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
path_sono = os.path.join('sintetizados', 'pruea_ruidosos')
#path_audio = os.path.join('nuevos', 'audios')
#path_sono = path_audio = 'filtro'

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
uoch, uolb, uolg, rb, rdis = (250/2.0)*1e8, 0.1 , 1/35., 1e7, 1e4   # 24*10000 , y con 350/3.0, la frec de la oec en 4000 Hz
fsamp, L = 882000.0, 0.045
dt = 1/fsamp
tiempo_total = 0.95
N=int((L/(350*dt))//1) #lo que tarda la onda en ir y reflejarse en el tracto

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
        l = int(d/dt)
        i2= i-l   
        k = np.arange(i2, j)
        
        phi = (alphai * j - alphaf * i) / dj
        omega = (alphaf - alphai) / dj
        freqs[i2:j] = media + amplitud * np.sin(phi + omega * k )
        beta[i2:j] = .5

def senpol(ti, tf, media, amplitud, alphai, alphaf, grado,
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
        l = int(d/dt)
        i2= i-l   
        k = np.arange(i2, j)
        
        phi = (alphai * j - alphaf * i) / dj
        omega = (alphaf - alphai) / dj
        freqs[i2:j] = media + amplitud * (np.sin(phi + omega * k ) ** grado)
        beta[i2:j] = .5

tiempos_steps = np.array([0.098, 0.122, 0.01, 0.154, 0.019, 0.03, 0.026, 0.046, 0.329])

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

    ### benteveo_BVRoRo_highpass_notch
    f = 1
    
    tiempos = np.cumsum(tiempos_steps + normal(0, 0.01, tiempos_steps.shape))
    
#    ti1 = max(0.098 + normal(0, 0.04), 0.01) #para que no sea negativo
#    medio1 = (ti1 + 0.122) * normal(1, 0.07)
#    tf1 = medio1 + 0.01 * normal(1, 0.1)
    senito(ti=tiempos[0], tf=tiempos[1], media=-70+normal(0,80), amplitud=1800, alphai=2.44, alphaf=0.7,
          f=f*1.6, freqs=frecuencias, beta=beta, amps=amplitudes, param=2, d=0.05, 
          fin=False)
    frec_final = frecuencias[int(tiempos[1]/dt)]
    rectas(ti=tiempos[1], tf=tiempos[2], wi=frec_final, wf=frec_final-310*normal(0,0.1),
          f=f*1.3, freqs=frecuencias, beta=beta, amps=amplitudes, 
          inicio=False)
    
    #A: opcion 2 (mejor)
#    ti2 = tf1 + 0.154 * normal(1, 0.06)
#    medio21 = ti2 + 0.019 * normal(1, 0.1)
#    medio22 = medio21 + 0.03 * normal(1, 0.1)
#    tf2 = medio22 + 0.026 * normal(1, 0.1)
    senito(tiempos[3], tf=tiempos[4], media=-900, amplitud=2530, alphai=2.35, alphaf=1.29,
          f=f*1.5, freqs=frecuencias, beta=beta, amps=amplitudes, param=2, d=0.05, 
          fin=False)
    frec_final = frecuencias[int(tiempos[4]/dt)]
    expo(ti=tiempos[4], tf=tiempos[5], wi=frec_final, wf=frec_final-430*normal(1, 0.5), tau=3.4,
        f=f, freqs=frecuencias, beta=beta, amps=amplitudes, 
        inicio=False, fin=False)
    senito(ti=tiempos[5], tf=tiempos[6], media=-1260, amplitud=2380, alphai=7.8, alphaf=7.3,
          f=f, freqs=frecuencias, beta=beta, amps=amplitudes, 
          inicio=False)
        
    #B: opcion 3 (mejor)
#    ti3 = tf2 + 0.046 * normal(1, 0.1)
#    tf3 = ti3 + 0.329 * normal(1, 0.04)
    senpol(ti=tiempos[7], tf=tiempos[8],
           media=-7300*normal(1, 0.1), amplitud=8700, alphai=1.86, alphaf=1.22,
          grado=1, f=f, freqs=frecuencias, beta=beta, amps=amplitudes, param=2, d=0.03)      
    
    corrimiento = 200 + normal(0, 15)
    donde = frecuencias != 0
    frecuencias[donde] = frecuencias[donde] - corrimiento

    #ploteo los parámetros
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
    
    kappa_todos = 6.56867694e-08 * frecuencias*frecuencias + 4.23116382e-05 * frecuencias + 2.67280260e-02
    kappa_todos *= normal(1,0.2,cant_puntos)
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
    sonido += normal(0, 7e-4, len(sonido))
    
    f, t, Sxx = signal.spectrogram(sonido, fsamp, window=('gaussian',20*128),
                                   nperseg=10*1024, noverlap=18*512, scaling='spectrum')
    
    fig, ax = plt.subplots()
    ax.pcolormesh(t,f,np.log10(Sxx),rasterized=True,
                  cmap=plt.get_cmap('Greys'))
    ax.set_ylim(10,8000)
    ax.axis('off')
#    ax.grid()
    fig.subplots_adjust(bottom = 0, top = 1, left = 0, right = 1) #para que no tenga bordes blancos
    
    
    nombre = creo_nombre(path_sono, nombre_base, '.jpg')
    fig.savefig(nombre, dpi=100)
#    plt.close()
    
#    scaled = (sonido/np.max(np.abs(sonido))).astype(np.float32)
#    nombre = creo_nombre(path_audio, nombre_base, '.wav')
#    write(nombre, int(fsamp/20), scaled[::20])
    
    
    print('listo {} de {}!'.format(i+1, cant_sintesis))
#    print('\a') #sonido al final de la integración
    
        
