# -*- coding: utf-8 -*-
"""
Genero sonogramas de los archivos de audio
"""

from scipy.io import wavfile
from scipy.signal import spectrogram 
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import os
from skimage import filters #para otsu


audioPath = '/home/marcos/Documents/Audios/Finches'
audioFiles = os.listdir(audioPath) #nombres de los archivos de audio
audioViejo = audioFiles.copy() #en el orden viejo (antes de ordenar alfabeticamente)
audioFiles.sort() #nombres ordenados

#%% Duración de cada archivo:

#calculo duracion de un archivo
def duracion(path,file):
    sf,data = wavfile.read(path + '/' + file)
    return len(data)/sf
    
dur = []
for file in audioFiles:
    dur.append(duracion(audioPath,file))

print('Duración media: {:.3f}'.format(np.mean(dur)))
print('Duración máxima: {:.3f}'.format(np.max(dur)))
print('Duración mínima: {:.3f}'.format(np.min(dur)))

#%% Un sonograma (ESTOY USANDO LOS DE PRAAT)
cual = 0;
sf,data = wavfile.read(audioPath+'/'+audioFiles[cual])

f, t, son = spectrogram(data, sf)
plt.pcolormesh(t, f, son)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()


#%% Las ya hechas con duraciones incorrectas:

hechas = ['zfsb-VV-BOS.png',
 'zfsb-VeNe-BOS.png',
 'bk93-BOS-3m.png',
 'con06_zfAB004-ni_20160630121420.png',
 'con17_zfAB012-bn_20170327093808_BOS01.png',
 'zfch002-zi_BOS2.png',
 'zfsb-NeRo-BOS.png',
 'con09_zfMA-zv_20160517061208.png',
 'zfch002-zi_BOS1.png',
 'zfsb-AV-BOS.png',
 'con13_zfAB009-ba_20161215075940_BOS01.png',
 'r38-BOS-3m.png',
 'zfmrb-BOS2.png',
 'zfsb-RoRo-BOS.png',
 'zfmri02-BOS2.png',
 'zfsb-FuVe-BOS.png',
 'zfsb-CC-BOS.png',
 'con04_zfAB002-rn_20160510061624.png',
 'con07_zfAS01-nb_20160527060850.png',
 'con03_zfCH002-zi_20160604095230.png',
 'con14_zfAB006-ia_20161121074139_BOS01.png']


#%% Import y trabajo sonogramas

sonoPath = '/home/marcos/Documents/Codigos/Analisis cantos/Sonogramas'
sonoFiles = os.listdir(sonoPath)
sonoFiles.sort()

escala = 1000 #cantidad de pixeles del sonograma por segundo
cual = 41 #qué figura miro

#cargo imagen, transformo a escala de grises
sono1 = Image.open(sonoPath + '/' + sonoFiles[cual]).convert('L')
#recorto la parte que me interesa (dentro de los ejes)
sono1 = sono1.crop((360, 238, 6845, 1860))
#reescalo la imagen para que el tamaño refleje la duración
#convierto a np.array
alto = sono1.size[1]
ancho = int(dur[cual] * escala)
A = np.array(sono1.resize((ancho,alto),Image.LANCZOS))
A = 255-A #la escala está al revés (255 es poca potencia)

intens = np.mean(A,axis=0) #intensidad media en func. del tiempo
intens /= np.max(intens) #normalizo
tiempo = np.linspace(0,dur[cual],intens.size)
edges = np.diff(intens) #los puntos donde emiezan y terminan
edges /= np.max(edges) #normalizo
cuadrado = intens>0.25
difcuadrado = np.diff(cuadrado)

#intento bajar ruido con otsu thresholding
thresh = filters.threshold_otsu(A) #otsu
B = A.copy()
B[B<thresh] = 0
threshInt = np.mean(B,axis=0) #calcula la intensidad "sin ruido"
threshInt /= np.max(threshInt) #normalizo
threshCu = threshInt>0.25
diffThreshCu = np.diff(threshCu)

#Plots:
fig = plt.figure(1,figsize=[18.39,  9.27])
#a0 = plt.subplot(2,1,1)
plt.imshow(A,cmap='Greys',extent=[0, dur[cual],0, .5])
#plt.plot(tiempo[:-1],difcuadrado*0.5,'r')
plt.plot(tiempo[:-1],diffThreshCu*0.5,'g--')
tit_obj = plt.title('{}: {}, duración:{:.1f}s'.format(cual,sonoFiles[cual],dur[cual]))
if sonoFiles[cual] in hechas:
    plt.setp(tit_obj,color='r')
    iViejo = hechas.index(sonoFiles[cual]) #indice en el orden viejo (antes de ordenar)
    plt.setp(tit_obj,text='{}: VIEJO:{}, duración:{:.1f}s, factor = {:.2f}'.format(
    cual,iViejo,dur[cual],dur[cual]/duracion(audioPath,audioViejo[iViejo])))
    
#a1 = plt.subplot(2,1,2,sharex=a0)
#plt.plot(tiempo,intens)
#plt.plot(tiempo,cuadrado,'r')
#plt.plot(tiempo,threshCu,'g--')