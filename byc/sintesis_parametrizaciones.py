--- BENTEVEO ---

###Original (Gabo)
senito(0.166,0.32+0.05*normal(1,0.1),1310*0.5*normal(1,0.1),200*0.5*normal(1,0.1),0,np.pi,0.7*1.1,frecuencias,beta,amplitudes)
senito(0.58,0.7,1305*0.5*normal(1,0.1),600*0.5*normal(1,0.1),-np.pi/4.0,3*np.pi/2.0,0.7*1,frecuencias,beta,amplitudes)
senito(0.74+0.05*normal(1,0.1),1.06,1301,200*normal(1,0.05),0,np.pi+np.pi/4.0,0.7*1,frecuencias,beta,amplitudes)

### Original (Marcos)
f = 0.35
senito(ti=0.184, tf=0.33, media=1750, amplitud=70, alphai=2.4, alphaf=0.7,
      f=f, freqs=frecuencias, beta=beta, amps=amplitudes, param=2, d=0.05)

senito(ti=0.59, tf=0.64, media=-870, amplitud=2960, alphai=2.35, alphaf=1.34,
      f=f, freqs=frecuencias, beta=beta, amps=amplitudes, param=2, d= 0.05, fin=False)
expo(ti=0.64, tf=0.69, wi=2010, wf=160, tau=0.68,
    f=f, freqs=frecuencias, beta=beta, amps=amplitudes, inicio=False)

senito(ti=0.737, tf=1.054, media=1290, amplitud=570, alphai=9.7, alphaf=6,
      f=f, freqs=frecuencias, beta=beta, amps=amplitudes, param=2, d = 0.03)

senito(0.166,0.32+0.05,1310*0.5,200*0.5,0,np.pi,0.7*1.1,frecuencias,beta,amplitudes)
senito(0.58,0.7,1305*0.5,600*0.5,-np.pi/4.0,3*np.pi/2.0,0.7*1,frecuencias,beta,amplitudes)
senito(0.74+0.05,1.06,1301,200,0,np.pi+np.pi/4.0,0.7*1,frecuencias,beta,amplitudes)

### benteveo_BVRoRo_highpass_notch
f = 1
senito(ti=0.098, tf=0.22, media=-70, amplitud=1800, alphai=2.44, alphaf=0.7,
      f=f, freqs=frecuencias, beta=beta, amps=amplitudes, param=2, d=0.05, 
      fin=False)
rectas(ti=0.22, tf=0.23, wi=1100, wf=790,
      f=f, freqs=frecuencias, beta=beta, amps=amplitudes, 
      inicio=False)

#A: opcion 1
senito(ti=0.384, tf=0.422, media=-900, amplitud=2530, alphai=2.35, alphaf=1.29,
      f=f, freqs=frecuencias, beta=beta, amps=amplitudes, param=2, d=0.05, 
      fin=False)
expo(ti=0.422, tf=0.476, wi=1530, wf=700, tau=2.1,
    f=f, freqs=frecuencias, beta=beta, amps=amplitudes, 
    inicio=False)
#A: opcion 2 (mejor)
medio1, medio2 = 0.422, 0.45
senito(ti=0.384, tf=medio1, media=-900, amplitud=2530, alphai=2.35, alphaf=1.29,
      f=f, freqs=frecuencias, beta=beta, amps=amplitudes, param=2, d=0.05, 
      fin=False)
expo(ti=medio1, tf=medio2, wi=1530, wf=1100, tau=3.4,
    f=f, freqs=frecuencias, beta=beta, amps=amplitudes, 
    inicio=False, fin=False)
senito(ti=medio2, tf=0.476, media=-1260, amplitud=2380, alphai=7.8, alphaf=7.3,
      f=f, freqs=frecuencias, beta=beta, amps=amplitudes, 
      inicio=False)

#B: opcion 1
senito(ti=0.582, tf=0.831, media=-410, amplitud=1900, alphai=2.44, alphaf=0.7,
      f=f, freqs=frecuencias, beta=beta, amps=amplitudes, param=2, d=0.05)
#B: opcion 2 (mejor)
senito(ti=0.582-0.06, tf=0.831+0.02, media=-7300, amplitud=8700, alphai=1.86, alphaf=1.22,
      f=f, freqs=frecuencias, beta=beta, amps=amplitudes, param=2, d=0.03)           

corrimiento = 200
donde = frecuencias != 0
frecuencias[donde] = frecuencias[donde] - corrimiento

###benteveo_XC433508_highpass_notch
f=2
senito(ti=0.067, tf=0.153, media=-500, amplitud=2600, alphai=2.44, alphaf=0.7,
      f=f, freqs=frecuencias, beta=beta, amps=amplitudes, param=2, d=0.05
      )

senito(ti=0.417, tf=0.457, media=-650, amplitud=2570, alphai=2.35, alphaf=1.34,
      f=f, freqs=frecuencias, beta=beta, amps=amplitudes, param=2, d=0.05, 
      fin=False)
expo(ti=0.457, tf=0.522, wi=1850, wf=1430, tau=3.8,
      f=f, freqs=frecuencias, beta=beta, amps=amplitudes, 
      inicio=False, fin=False)
senito(ti=0.522, tf=0.555, media=700, amplitud=740, alphai=7.8, alphaf=6.9,
      f=f, freqs=frecuencias, beta=beta, amps=amplitudes, 
      inicio=False)

rectas(ti=0.654, tf=0.686, wi=1110, wf=1890,
      f=f, freqs=frecuencias, beta=beta, amps=amplitudes,
      fin=False)
senito(ti=0.686, tf=0.788, media=1830, amplitud=100, alphai=2.44, alphaf=0.7,
      f=f, freqs=frecuencias, beta=beta, amps=amplitudes, 
      inicio=False, fin=False)
senito(ti=0.788, tf=0.801, media=-800, amplitud=2700, alphai=1.5, alphaf=1.1,
      f=f, freqs=frecuencias, beta=beta, amps=amplitudes,
      inicio=False, fin=False)
senito(ti=0.801, tf=0.955, media=1570, amplitud=40, alphai=1.5, alphaf=0.1,
      f=f, freqs=frecuencias, beta=beta, amps=amplitudes,
      inicio=False, fin=False)
senito(ti=0.953, tf=0.986, media=-27420, amplitud=29000, alphai=1.55, alphaf=1.39,
      f=f, freqs=frecuencias, beta=beta, amps=amplitudes, 
      inicio=False)

# Filtro
rectas(0.01, tiempo_total-0.01, 200, 5000,
       f=1, freqs=frecuencias, beta=beta, amps=amplitudes,
       inicio=False, fin=False)

nombre_base = 'uoch={:.2e}_uolb={}_uolg={:.2f}'.format(uoch, uolb, uolg)

--- CHINGOLO ---
#%%%# El resto están en el cuaderno, buscarlas ahí. #%%%#

###Original (Gabo)
rectas(0.03,0.42,4529,3200,0.15,frecuencias,beta,amplitudes)
rectas(0.535,0.774+0.01,3990,4300,1,frecuencias,beta,amplitudes)

tiempito=1.05
senito(0.863,tiempito,6500,900,-np.pi/2.0-np.pi/4.,1*np.pi,1,frecuencias,beta,amplitudes)
expo(tiempito,1.288,6500+500,3460+100,1.0,frecuencias,beta,amplitudes)

fmax=5900+100*normal(0,1)
senito(1.33,1.41+0.0005*normal(0,1),fmax,2740,np.pi,3*np.pi/2.0+np.pi/16.0,1,frecuencias,beta,amplitudes)
senito(1.42+0.0005*normal(0,1),1.48,fmax,2740,np.pi,3*np.pi/2.0+np.pi/16.0,1,frecuencias,beta,amplitudes)

tiempin=0.0005*normal(0,1)
senito(1.49+tiempin,1.55+tiempin,fmax,2740,np.pi,3*np.pi/2.0+np.pi/16.0,1,frecuencias,beta,amplitudes)
senito(1.57+tiempin,1.62+tiempin,fmax,2740,np.pi,3*np.pi/2.0+np.pi/16.0,1,frecuencias,beta,amplitudes)
senito(1.64+tiempin,1.70+tiempin,fmax,2740,np.pi,3*np.pi/2.0+np.pi/16.0,1,frecuencias,beta,amplitudes)
senito(1.72+tiempin,1.77+tiempin,fmax,2740,np.pi,3*np.pi/2.0+np.pi/16.0,.9,frecuencias,beta,amplitudes)
senito(1.79+tiempin,1.84+tiempin,fmax,2740,np.pi,3*np.pi/2.0+np.pi/16.0,.7,frecuencias,beta,amplitudes)
senito(1.87+tiempin,1.92+tiempin,fmax,2740,np.pi,3*np.pi/2.0,.6,frecuencias,beta,amplitudes)


###Chingolo_XC462515_denoised
f = .5
rectas(ti=0.086, tf=0.168, wi=4560-300, wf=4711-300, 
       f=f, freqs=frecuencias, beta=beta, amps=amplitudes, param=2, d=0.03)

expo(ti=0.315, tf=0.569, wi=4260, wf=4030, tau=-1.5,
       f=f, freqs=frecuencias, beta=beta, amps=amplitudes, param=2, d=0.03)

medio=0.729
rectas(ti=0.677, tf=medio, wi=6030, wf=5730,
     f=f, freqs=frecuencias, beta=beta, amps=amplitudes, param=2, d=0.03,
     fin=False)
expo(ti=medio, tf=0.961, wi=5736, wf=1370, tau=0.8,
       f=f, freqs=frecuencias, beta=beta, amps=amplitudes, inicio=False)

deltat, t0, t1 = 0.0028, 1.08, 1.124
paso = deltat + t1 - t0
for k in range(7):
#        rectas(t0 + paso*k, t1 + paso*k, 6945, 3839,
#               f=1, freqs=frecuencias, beta=beta, amps=amplitudes)
    rectas(t0 + paso*k, t1 + paso*k, 7030, 3760, 
         f=1, freqs=frecuencias, beta=beta, amps=amplitudes)

--- ZORZAL ---
#%%%#

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