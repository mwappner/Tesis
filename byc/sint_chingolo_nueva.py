# -*- coding: utf-8 -*-
"""
Created on march 2017

@author: Gabo Mindlin

Integrator with rk4, and tube with delays

it creates wav


"""
import os

cant_sintesis = 1 #cuantos cantos voy a sintetizar
nombre_base = 'chingolo' #nombre de los sonogramas
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

gamma=24000
uoch, uolb, uolg, rb, rdis = (350/5.0)*100000000, 0.0001 , 1/20., 0.5*10000000, 24*10000 # 24*10000 , y con 350/3.0, la frec de la oec en 4000 Hz
fsamp, L=  882000.0, 0.025
dt = 1/fsamp
tiempo_total=2.1
N=int((L/(350*dt))//1)

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


def expo(ti, tf, wi, wf, f, freqs, beta, amps, tau=3):
    
    i=np.int(ti/dt)
    j=np.int(tf/dt)
    k = np.arange(j-i) / (j-i) # =  t - ti / (ti - tf)

    freqs[i:j] = wf + (wi-wf) * np.exp(-tau * k)
    beta[i:j] = 1
    amps[i:j] = f * np.sin(np.pi * k)

def rectas(ti, tf, wi, wf, f, freqs, beta, amps):
    
    i=np.int(ti/dt)
    j=np.int(tf/dt)
    k = np.arange(j-i) / (j-i) # =  t - ti / (ti - tf)

    freqs[i:j] = wi + (wf-wi) * k
    beta[i:j] = .5
    amps[i:j] = f * np.sin(np.pi * k)

def senito(ti, tf, media, amplitud, alphai, alphaf, f, freqs, beta, amps):
    
    i=np.int(ti/dt)
    j=np.int(tf/dt)
    k = np.arange(j-i) / (j-i) # =  t - ti / (ti - tf)
    
    freqs[i:j] = media + amplitud * np.sin(alphai + (alphaf - alphai) * k)
    beta[i:j] = .5
    amps[i:j] = f * np.sin(np.pi * k)

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
    
    
    
#    rectas(0.03,0.42,4529,3200,0.15,frecuencias,beta,amplitudes)
#    rectas(0.535,0.774+0.01,3990,4300,1,frecuencias,beta,amplitudes)
#    
#    tiempito=1.05
#    senito(0.863,tiempito,6500,900,-np.pi/2.0-np.pi/4.,1*np.pi,1,frecuencias,beta,amplitudes)
#    expo(tiempito,1.288,6500+500,3460+100,1.0,frecuencias,beta,amplitudes)
#    
#    fmax=5900+100*normal(0,1)
#    senito(1.33,1.41+0.0005*normal(0,1),fmax,2740,np.pi,3*np.pi/2.0+np.pi/16.0,1,frecuencias,beta,amplitudes)
#    senito(1.42+0.0005*normal(0,1),1.48,fmax,2740,np.pi,3*np.pi/2.0+np.pi/16.0,1,frecuencias,beta,amplitudes)
#    
#    tiempin=0.0005*normal(0,1)
#    senito(1.49+tiempin,1.55+tiempin,fmax,2740,np.pi,3*np.pi/2.0+np.pi/16.0,1,frecuencias,beta,amplitudes)
#    senito(1.57+tiempin,1.62+tiempin,fmax,2740,np.pi,3*np.pi/2.0+np.pi/16.0,1,frecuencias,beta,amplitudes)
#    senito(1.64+tiempin,1.70+tiempin,fmax,2740,np.pi,3*np.pi/2.0+np.pi/16.0,1,frecuencias,beta,amplitudes)
#    senito(1.72+tiempin,1.77+tiempin,fmax,2740,np.pi,3*np.pi/2.0+np.pi/16.0,.9,frecuencias,beta,amplitudes)
#    senito(1.79+tiempin,1.84+tiempin,fmax,2740,np.pi,3*np.pi/2.0+np.pi/16.0,.7,frecuencias,beta,amplitudes)
#    senito(1.87+tiempin,1.92+tiempin,fmax,2740,np.pi,3*np.pi/2.0,.6,frecuencias,beta,amplitudes)
    
    
    rectas(ti=0.0457, tf=0.2975, wi=4568, wf=4325, 
           f=0.15, freqs=frecuencias, beta=beta, amps=amplitudes)
    rectas(ti=0.2975, tf=0.581, wi=4757, wf=4757, 
           f=1, freqs=frecuencias, beta=beta, amps=amplitudes)
    
    medio1=0.745
    medio2=0.778
#    #opción 1:
#    expo(ti=0.633, tf=medio1, wi=2947, wf=1250, tau=-0.7,
#         f=1, freqs=frecuencias, beta=beta, amps=amplitudes)
#    senito(ti=medio1, tf=medio2, media=-930, amplitud=5800 ,alphai=1.83, alphaf=1.31,
#           f=1, freqs=frecuencias, beta=beta, amps=amplitudes)
#    #opción 2:
#    senito(ti=0.633, tf=medio1, media=18400, amplitud=-15500, alphai=1.51, alphaf=1.09,
#         f=1, freqs=frecuencias, beta=beta, amps=amplitudes)
#    senito(ti=medio1, tf=medio2, media=-930, amplitud=5800 ,alphai=1.83, alphaf=1.31,
#           f=1, freqs=frecuencias, beta=beta, amps=amplitudes)
    #opcion 3:
    senito(ti=0.633, tf=medio2, media=3900, amplitud=-970, alphai=1.8, alphaf=5.3,
           f=1, freqs=frecuencias, beta=beta, amps=amplitudes)
    
    expo(ti=medio2, tf=0.961, wi=4677, wf=2170, tau=0.8,
         f=1, freqs=frecuencias, beta=beta, amps=amplitudes)
    
    deltat, t0, t1 = 0.0028, 1.044, 1.102
    paso = deltat + t1 - t0
    for k in range(13):
#        rectas(t0 + paso*k, t1 + paso*k, 6945, 3839,
#               f=1, freqs=frecuencias, beta=beta, amps=amplitudes)
        expo(t0 + paso*k, t1 + paso*k, 6945, 3700, tau=3,
             f=1, freqs=frecuencias, beta=beta, amps=amplitudes)
    
#    tiempo = np.linspace(0, tiempo_total, cant_puntos)
#    plt.subplot(311)
#    plt.plot(tiempo[::10],frecuencias[::10], '.')
#    plt.subplot(312)
#    plt.plot(tiempo[::10],amplitudes[::10], '.')
#    plt.subplot(313)
#    plt.plot(tiempo[::10],beta[::10], '.')
#    
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
    sonido += 20*normal(0, .007, cant_puntos)
    
    f, t, Sxx = signal.spectrogram(sonido,fsamp,window=('gaussian',20*128),
                                   nperseg=10*1024,noverlap=18*512,scaling='spectrum')
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
    
    print('listo!')
    print('\a') #sonido al final de la integración
    

