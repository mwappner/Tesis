# -*- coding: utf-8 -*-
"""
Created on march 2017

@author: Gabo Mindlin

Integrator with rk4, and tube with delays

it creates wav


"""


import numpy as np
#import pylab      
from scipy.io.wavfile import write
from numpy.random import normal
from scipy import signal
import matplotlib.pyplot as plt
import os
from utils import new_name


global kappa
global feedback1 
global estimulo1
global destimulodt1
global b
    
    



for lazo in range(1):
    gamma=24000
    uoch, uolb, uolg, rb, rdis = (350/5.0)*100000000, 0.0001 , 1/20., 0.5*10000000, 24*10000 # 24*10000 , y con 350/3.0, la frec de la oec en 4000 Hz
    beta, dt, t0, tf, L= -0.15, 1/882000.0, 0, 0.5, 0.025
    t=0
    fsamp=1/dt
    tiempo_total=2.1
    frequencias=np.zeros(np.int(tiempo_total/(dt)))
    tiempos=np.zeros(np.int(tiempo_total/(dt)))
    k=np.zeros(np.int(tiempo_total/(dt)))
    amplitudes=np.zeros(np.int(tiempo_total/(dt)))
    beta=np.zeros(np.int(tiempo_total/(dt)))



    for i in range(np.int(tiempo_total/(dt))):
        tiempos[i]=i*dt
        beta[i]=-1.0
        amplitudes[i]=0.
        k[i]=10.

    v=np.zeros(5)
    v[0], v[1], v[2], v[3], v[4] =0.01,0.001,0.001, 0.0001, 0.0001
# --------------------
# Function definitions
# --------------------
    def ecuaciones(v, dv):
        x,y,i1,i2,i3 = v
        dv[0]=y
        dv[1]=-gamma*gamma*kappa*x-gamma*x*x*y+b*gamma*y
        dv[2]= i2
        dv[3]=-uolg*uoch*i1-(rdis*uolb+rdis*uolg)*i2+(uolg*uoch-rdis*rb*uolg*uolb)*i3+uolg*destimulodt+rdis*uolg*uolb*estimulo
        dv[4]=-(uolb/uolg)*i2-rb*uolb*i3+uolb*estimulo
        return dv


    
    def expo(ti,tf,wi,wf,factor,frequencias,beta,amplitudes):
        i=np.int(ti/dt)
        j=np.int(tf/dt)
        for k in range((j-i)):
            t=ti+k*dt
            frequencias[i+k]=wf+(wi-wf)*np.exp(-3*(t-ti)/((tf-ti)))
            beta[i+k]=1.0
            #amplitudes1[i+k]=(1/(1+np.exp(-(t-ti)/0.01))-1/(1+np.exp(-(t-tf)/0.01)))
            amplitudes[i+k]=factor*np.sin(np.pi*k/(j-i))
        return frequencias,beta,amplitudes


    def rectas(ti,tf,wi,wf,factor,frequencias,beta,amplitudes):
        i=np.int(ti/dt)
        j=np.int(tf/dt)
        for k in range((j-i)):
            t=ti+k*dt
            frequencias[i+k]=wi+(wf-wi)*(t-ti)/(tf-ti)
            beta[i+k]=1.0
            #amplitudes1[i+k]=(1/(1+np.exp(-(t-ti)/0.01))-1/(1+np.exp(-(t-tf)/0.01)))
            amplitudes[i+k]=factor*np.sin(np.pi*k/(j-i))
        return frequencias,beta,amplitudes

    def senito(ti,tf,media,amplitud,alphai,alphaf,factor,frequencias,beta,amplitudes):
        i=np.int(ti/dt)
        j=np.int(tf/dt)
        for k in range((j-i)):
            t=ti+k*dt
            frequencias[i+k]=media+amplitud*np.sin(alphai+(alphaf-alphai)*(t-ti)/(tf-ti))
            beta[i+k]=1.0
            #amplitudes1[i+k]=(1/(1+np.exp(-(t-ti)/0.01))-1/(1+np.exp(-(t-tf)/0.01)))*frequencias1[i+k]
            amplitudes[i+k]=factor*np.sin(np.pi*k/(j-i))
        return frequencias,beta,amplitudes
    
#genero el canto:
    rectas(0.03,0.42+0.01*normal(0,1),4529+500*normal(0,1),3200+500*normal(0,1),0.15,frequencias,beta,amplitudes)
    rectas(0.535,0.774+0.01*normal(0,1),3990+500*normal(0,1),4300+500*normal(0,1),1,frequencias,beta,amplitudes)
    tiempito=1.05+0.01*normal(0,1)
    senito(0.863,tiempito,6500+500*normal(0,1),900,-np.pi/2.0-np.pi/4.,1*np.pi,1,frequencias,beta,amplitudes)
    expo(tiempito,1.288,6500+500*normal(0,1),3460+100*normal(0,1),1.0,frequencias,beta,amplitudes)
    fmax=5900+100*normal(0,1)
    senito(1.33,1.41+0.0005*normal(0,1),fmax,2740,np.pi,3*np.pi/2.0+np.pi/16.0,1,frequencias,beta,amplitudes)
    senito(1.42+0.0005*normal(0,1),1.48,fmax,2740,np.pi,3*np.pi/2.0+np.pi/16.0,1,frequencias,beta,amplitudes)
    tiempin=0.0005*normal(0,1)
    senito(1.49+tiempin,1.55+tiempin,fmax,2740,np.pi,3*np.pi/2.0+np.pi/16.0,1,frequencias,beta,amplitudes)
    senito(1.57+tiempin,1.62+tiempin,fmax,2740,np.pi,3*np.pi/2.0+np.pi/16.0,1,frequencias,beta,amplitudes)
    senito(1.64+tiempin,1.70+tiempin,fmax,2740,np.pi,3*np.pi/2.0+np.pi/16.0,1,frequencias,beta,amplitudes)
    senito(1.72+tiempin,1.77+tiempin,fmax,2740,np.pi,3*np.pi/2.0+np.pi/16.0,.9,frequencias,beta,amplitudes)
    senito(1.79+tiempin,1.84+tiempin,fmax,2740,np.pi,3*np.pi/2.0+np.pi/16.0,.7,frequencias,beta,amplitudes)
    senito(1.87+tiempin,1.92+tiempin,fmax,2740,np.pi,3*np.pi/2.0,.6,frequencias,beta,amplitudes)
#    senito(1.96+tiempin,2.+tiempin,fmax,2740,np.pi,3*np.pi/2.0-np.pi/16.0,.6,frequencias,beta,amplitudes)

    
    

#%%


    def rk4(dv,v,n,t,dt):
        v1=[]
        k1=[]
        k2=[]
        k3=[]
        k4=[]
        for x in range(0, n):
            v1.append(x)
            k1.append(x)
            k2.append(x)
            k3.append(x)
            k4.append(x)
            
        dt2=dt/2.0
        dt6=dt/6.0
        for x in range(0, n):
            v1[x]=v[x]
        dv(v1, k1)
        for x in range(0, n):
            v1[x]=v[x]+dt2*k1[x]
        dv(v1, k2)     
        for x in range(0, n):
            v1[x]=v[x]+dt2*k2[x]
        dv(v1, k3)
        for x in range(0, n):
            v1[x]=v[x]+dt*k3[x]
        dv(v1, k4)
        for x in range(0, n):
            v1[x]=v[x]+dt*k4[x]        
        for x in range(0, n):
            v[x]=v[x]+dt6*(2.0*(k2[x]+k3[x])+k1[x]+k4[x])
        return v


# una integracion

    n=5 #Cantidad de variables   
    x1=[]
    y1=[]
    tiempo1=[]
    sonido=[]
    sonido_total=[]
    amplitud1=[]
    forzado1=[]
    dforzadodt1=[]
    elbeta1=[]
    
    cont1=0
    N=int((L/(350*dt))//1)
    fil1=np.zeros(N)
    back1=np.zeros(N)
    feedback1=0

#while t<tf:
    print('integrando...')
    for i in range(np.int(tiempo_total/(dt))):
        kappa=6.56867694e-08*frequencias[i]*frequencias[i]+4.23116382e-05*frequencias[i]+2.67280260e-02
        b=beta[i]*(1+normal(0,0.05))
        t=i*dt
        estimulo=fil1[N-1]
        destimulodt=(fil1[N-1]-fil1[N-2])/dt
        rk4(ecuaciones,v,n,t,dt)
        fil1[0]=v[1]+back1[N-1]
        back1[0]=-0.65*fil1[N-1]
        fil1[1:]=fil1[:-1]
        back1[1:]=back1[:-1]
        feedback1=back1[N-1]
        x1.append(cont1)  #ACÁ ARMO LOS ARREGLOS DE X Y Z CON LOS RESULTADOS QUE VA LARGANDO "V"
        y1.append(cont1)
        tiempo1.append(cont1)
        sonido.append(cont1)
        sonido_total.append(cont1)
        amplitud1.append(cont1)
        forzado1.append(cont1)
        dforzadodt1.append(cont1)
        elbeta1.append(cont1)
        x1[cont1]=v[0]
        y1[cont1]=v[1]
        tiempo1[cont1]=t
        # sonido[cont]=back[0]
        sonido[cont1]=v[4]*amplitudes[i]*(1+normal(0,0.01))
        sonido_total[cont1]=0
        amplitud1[cont1]=amplitudes[i]*(1+normal(0,0.01))
        forzado1[cont1]=estimulo
        dforzadodt1[cont1]=destimulodt
        elbeta1[cont1]=beta[i]

        cont1=cont1+1
  

    for i in range(len(sonido)):
        sonido_total[i]=(sonido[i])*1000+20*normal(0,0.01)


#pylab.plot(tiempo3,sonido_total)
#pylab.xlabel(r'$t\;/\mathrm{sec}$')
#pylab.ylabel(r'$sonido total\;/\mathrm{arb. units}$')
#pylab.show() 
#%%



    sonido=np.asarray(sonido_total)  
 
    path_sono = 'pruebas_sintesis'
#    path_sono = os.path.join('sintetizados', 'sonogramas', 'train', 'Chingolos')
#    path_audio =  os.path.join('sintetizados', 'audios', 'train', 'Chingolos')
# 
    f, t, Sxx = signal.spectrogram(sonido,882000,window=('gaussian',20*128),nperseg=10*1024,noverlap=18*512,scaling='spectrum')
    plt.pcolormesh(t,f,np.log10(Sxx),rasterized=True,cmap=plt.get_cmap('Greys'))
#plt.pcolormesh(t,f,Sxx,cmap=plt.get_cmap('Greys'))
    plt.ylim(10,10000)
#plt.ylabel('Frequency [Hz]')
#plt.xlabel('Time [sec]')
    plt.axis('off')
    nombre = new_name(os.path.join(path_sono, 'chingolo.jpeg'))
    plt.savefig(nombre, dpi=50, facecolor='w', edgecolor='w',
                orientation='portrait', papertype=None, format=None,
                transparent=False, bbox_inches=None, pad_inches=0.1,
                frameon=None)
#    plt.show()
    plt.close()
#    scaled = np.int16(sonido/np.max(np.abs(sonido)) * 32767)
#    nombre = new_name(os.path.join(path_audio, 'test.wav'))
#    write(nombre, 882000, scaled)
    
    
    print('listo {}!'.format(lazo+1))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    