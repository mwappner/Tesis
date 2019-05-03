# -*- coding: utf-8 -*-
"""
Created on march 2017

@author: Gabo Mindlin

Integrator with rk4, and tube with delays

it creates wav


"""


import numpy as np
from numpy.random import normal
#import pylab      
#from scipy.io.wavfile import write
import random
from scipy import signal
import matplotlib.pyplot as plt
import os
from utils import new_name


#global kappa
#global feedback1 
global estimulo1
global destimulodt1
#global b

gamma = 24000
uoch, uolb, uolg, rb, rdis = (350/2.0)*100000000, 0.1 , 1/35., 10000000, 10000  # 24*10000 , y con 350/3.0, la frec de la oec en 4000 Hz
fsamp, t0, tf, L = 882000.0, 0, 0.5, 0.045
#t = 0
dt = 1/fsamp
tiempo_total = 1.2
cant_puntos = np.int(tiempo_total/(dt))

frequencias=np.zeros(cant_puntos)
cant_puntos = np.int(tiempo_total/(dt))
beta = np.ones(cant_puntos) * -.10
amplitudes = np.zeros(cant_puntos)
#k = np.ones(cant_puntos) * 10

v = np.array([0.01, 0.001, 0.001, 0.0001, 0.0001])


# Function definitions
# --------------------
def ecuaciones(v, kappa, b):
    x,y,i1,i2,i3 = v
    dv = np.array((
        y,
        -gamma*gamma*kappa*(x+0.1*x*x*x)-gamma*(x*x+0.3*x*x*x)*y+b*gamma*y,
        i2,
        -uolg*uoch*i1-(rdis*uolb+rdis*uolg)*i2+(uolg*uoch-rdis*rb*uolg*uolb)*i3+uolg*destimulodt+rdis*uolg*uolb*estimulo,
        -(uolb/uolg)*i2-rb*uolb*i3+uolb*estimulo))
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


def expo(ti,tf,wi,wf,factor,frequencias,beta,amplitudes):
    i=np.int(ti/dt)
    j=np.int(tf/dt)
    for k in range((j-i)):
        t=ti+k*dt
        frequencias[i+k]=wf+(wi-wf)*np.exp(-3*(t-ti)/((tf-ti)))
        beta[i+k]=.50
        #amplitudes1[i+k]=(1/(1+np.exp(-(t-ti)/0.01))-1/(1+np.exp(-(t-tf)/0.01)))
        amplitudes[i+k]=factor*np.sin(np.pi*k/(j-i))
    return frequencias,beta,amplitudes


def rectas(ti,tf,wi,wf,factor,frequencias,beta,amplitudes):
    i=np.int(ti/dt)
    j=np.int(tf/dt)
    for k in range((j-i)):
        t=ti+k*dt
        frequencias[i+k]=wi+(wf-wi)*(t-ti)/(tf-ti)
        beta[i+k]=.50
        #amplitudes1[i+k]=(1/(1+np.exp(-(t-ti)/0.01))-1/(1+np.exp(-(t-tf)/0.01)))
        amplitudes[i+k]=factor*np.sin(np.pi*k/(j-i))
    return frequencias,beta,amplitudes

def senito(ti,tf,media,amplitud,alphai,alphaf,factor,frequencias,beta,amplitudes):
    i=np.int(ti/dt)
    j=np.int(tf/dt)
    for k in range((j-i)):
        t=ti+k*dt
        frequencias[i+k]=media+amplitud*np.sin(alphai+(alphaf-alphai)*(t-ti)/(tf-ti))
        beta[i+k]=1.
        tau=(j-i)/5.0
        #amplitudes1[i+k]=(1/(1+np.exp(-(t-ti)/0.01))-1/(1+np.exp(-(t-tf)/0.01)))*frequencias1[i+k]
        amplitudes[i+k]=factor*(k/tau)*np.exp(-k/(tau))*(1+random.normalvariate(0,0.1))*(1+0.4*np.sin(2*np.pi*(k/6820.)))
        #amplitudes[i+k]=factor*np.sin(np.pi*k/(j-i))*(1+0.3*np.sin(2*np.pi*(k/6820.)))*(1+random.normalvariate(0,0.5))
    return frequencias,beta,amplitudes

#--------
#genero el el patrón de frecuencias y amplitudes que es el canto:
#inicio=np.abs(random.normalvariate(0,0.1))

senito(0.166,0.32+0.05*(1+random.normalvariate(0,0.1)),1310*(1+0.5*random.normalvariate(0,0.1)),200*(1+0.5*random.normalvariate(0,0.1)),0,np.pi,0.7*1.1,frequencias,beta,amplitudes)
senito(0.58,0.7,1305*(1+0.5*random.normalvariate(0,0.1)),600*(1+0.5*random.normalvariate(0,0.1)),-np.pi/4.0,3*np.pi/2.0,0.7*1,frequencias,beta,amplitudes)
senito(0.74+0.05*(1+random.normalvariate(0,0.1)),1.06,1301,200*(1+random.normalvariate(0,0.05)),0,np.pi+np.pi/4.0,0.7*1,frequencias,beta,amplitudes)
   


#-------
#Integro
v4 = []
#amplitud1=[]; forzado1=[]; dforzadodt1=[]; elbeta1=[]
#x1=[]; y1=[]

cont1 = 0
N=int((L/(350*dt))//1)
fil1 = np.zeros(N)
back1 = np.zeros(N)
#feedback1=0

print('integrando...')

kappa_todos = (6.56867694e-08 * frequencias**2 + 4.23116382e-05 * frequencias + 2.67280260e-02) * normal(1,0.2,cant_puntos)
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
#    feedback1 = back1[-1]
      
    v4.append(v[4])
#    x1.append(v[0])
#    y1.append(v[1])
#    amplitud1.append(amplitudes[i]*(1+random.normalvariate(0,0.2)))
#    forzado1.append(estimulo)
#    dforzadodt1.append(destimulodt)
#    elbeta1.append(beta[i])

sonido = np.array(v4) * amplitudes
sonido *= 1000
sonido += 20 * normal(0, .01, len(sonido))

path_sono = 'pruebas_sintesis'
#    path_sono = os.path.join('sintetizados', 'sonogramas', 'validation', 'Benteveos')
#    path_audio =  os.path.join('sintetizados', 'audios', 'validation', 'Benteveos')
 
f, t, Sxx = signal.spectrogram(sonido,882000,window=('gaussian',20*128),
                               nperseg=10*1024,noverlap=18*512,scaling='spectrum')
plt.pcolormesh(t,f,np.log10(Sxx),rasterized=True,cmap=plt.get_cmap('Greys'))
#plt.pcolormesh(t,f,Sxx,cmap=plt.get_cmap('Greys'))
plt.ylim(10,10000)
#plt.ylabel('Frequency [Hz]')
#plt.xlabel('Time [sec]')
plt.axis('off')
nombre = new_name(os.path.join(path_sono, 'benteveo.jpeg'))
plt.subplots_adjust(bottom = 0, top = 1, left = 0, right = 1)
plt.savefig(nombre, dpi=100)

#plt.savefig(nombre, dpi=50, facecolor='w', edgecolor='w',
#            orientation='portrait', papertype=None, format=None,
#            transparent=False, bbox_inches=None, pad_inches=0.1,
#            frameon=None)
#    plt.show()
plt.close()
#    scaled = np.int16(sonido/np.max(np.abs(sonido)) * 32767)
#    nombre = new_name(os.path.join(path_audio, 'test.wav'))
#    write(nombre, 882000, scaled)


print('listo!')
print('\a') #sonido al final de la integración

    
