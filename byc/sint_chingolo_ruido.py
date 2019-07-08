# -*- coding: utf-8 -*-
"""
Created on march 2017

@author: Gabo Mindlin

Integrator with rk4, and tube with delays

it creates wav


"""
import os

cant_sintesis = 2500 #cuantos cantos voy a sintetizar
nombre_base = 'chingolo' #nombre de los sonogramas
path_sono = os.path.join('sintetizados', 'sonogramas', 'chingolos')
path_audio = os.path.join('sintetizados', 'audios', 'chingolos')

import numpy as np
from numpy.random import normal

from scipy.io.wavfile import write
from scipy import signal

#para correr remotamente
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from utils import new_name, Testimado

creo_nombre = lambda path, base, formato: new_name(os.path.join(path, base + formato))
estimador = Testimado(cant_sintesis)
# --------------------
# Parámetros generales
# --------------------
global estimulo1
global destimulodt1

gamma=24000
uoch, uolb, uolg, rb, rdis = (350/5.0)*1e8, 1e-4, 1/20., 5e6, 2.4e5 # 2.4e5 , y con 350/3.0, la frec de la oec en 4000 Hz
fsamp, L=  882000.0, 0.025
dt = 1/fsamp
tiempo_total=1.8
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
    k = np.linspace(0,1,dj) # =  t - ti / (tf - ti)
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
    k = np.linspace(0,1,dj) # =  t - ti / (ti - tf)
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
    k = np.linspace(0,1,dj)
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

tiempos = 0.086, 0.168, 0.315, 0.569, 0.677, 0.729, 0.961, 1.08, 1.124
intervalos = np.diff(tiempos)

#%%
#reinicio valores
for i in range(cant_sintesis):
    
    frecuencias=np.zeros(cant_puntos)
    amplitudes=np.zeros(cant_puntos)
    beta = np.full(cant_puntos, -1.0)
    
    v = np.array([0.01, 0.001, 0.001, 0.0001, 0.0001])

    completos = np.array((max(tiempos[0] + normal(0, 0.02), 0.033), *intervalos))
    tiempos_ruidosos = np.cumsum(completos + normal(0, 0.007, completos.shape))
    
    # -----------------------------------
    # Genero los parámetros de los cantos
    # -----------------------------------
    
    ### Chingolo_XC462515_denoised
    f = .5
    f0 = 4260 *normal(1,0.1)
    rectas(tiempos[0], tiempos[1], wi=f0, wf=f0 + 150 * normal(1,0.1), 
           f=f, freqs=frecuencias, beta=beta, amps=amplitudes, param=2, d=0.03)
    
    f0 = 4260 *normal(1,0.1)
    expo(tiempos[2], tiempos[3], 
         wi=f0, wf=f0 - 230 * normal(1,0.01), tau=-1.5,
           f=f, freqs=frecuencias, beta=beta, amps=amplitudes, param=2, d=0.03)
    
    f0 = 6030 *normal(1,0.1)
    rectas(tiempos[4], tiempos[5], wi=f0, wf=f0 - 300 * normal(1,0.1),
         f=f, freqs=frecuencias, beta=beta, amps=amplitudes, param=2, d=0.03,
         fin=False)
    frec_final = frecuencias[int(tiempos[5]/dt)-1]
    expo(tiempos[5], tiempos[6], 
         wi=frec_final, wf=frec_final - 4366 * normal(1, 0.06), tau=0.8,
           f=f, freqs=frecuencias, beta=beta, amps=amplitudes, inicio=False)
    
    deltat, t0, t1 = 0.0028, tiempos[7], tiempos[8]
    paso = deltat + normal(0,0.0003) + t1 - t0
    fi = 7030 * normal(1, 0.03)
    ff = fi - 3270 * normal(1, 0.02)
    for k in range(7):
        rectas(t0 + paso*k, t1 + paso*k, fi * normal(1, 0.02), ff, 
             f=1, freqs=frecuencias, beta=beta, amps=amplitudes)

    
#    tiempo = np.linspace(0, tiempo_total, cant_puntos)
#    fig1, axs= plt.subplots(3,1, sharex=True)
#    axs[0].plot(tiempo[::10],frecuencias[::10], '.')
#    axs[1].plot(tiempo[::10],amplitudes[::10], '.')
#    axs[2].plot(tiempo[::10],beta[::10], '.')
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
        
    sonido = np.array(v4) * amplitudes
    sonido += normal(0, sonido.std()/2, len(sonido))
    
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
    plt.close(fig)
    
    scaled = (sonido/np.max(np.abs(sonido))).astype(np.float32)
    nombre = creo_nombre(path_audio, nombre_base, '.wav')
    write(nombre, int(fsamp/20), scaled[::20])
    
    print('listo {} de {}! ETA: {}'.format(i+1, cant_sintesis, estimador.time_str(i)))
#    print('\a') #sonido al final de la integración
    

