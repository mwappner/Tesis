# -*- coding: utf-8 -*-
"""
Genero sonogramas de los archivos de audio
"""

from scipy.io import wavfile
from scipy import signal as sg
import numpy as np
from matplotlib import pyplot as plt
import os


audioPath = '/home/marcos/Documents/Audios/Finches'
audioFiles = os.listdir(audioPath) #nombres de los archivos de audio
audioViejo = audioFiles.copy() #en el orden viejo (antes de ordenar alfabeticamente)
audioFiles.sort() #nombres ordenados

#%% Duración de cada archivo:

#calculo duracion de un archivo
def duracion(path,file):
    sf,data = wavfile.read(path + '/' + file)
    return len(data)/sf

#devuelve el ancho de la ventana actual
def Ancho(axes):
    xa = axes.get_xlim()
    return xa[1]-xa[0]

dur = []
for file in audioFiles:
    dur.append(duracion(audioPath,file))

print('Duración media: {:.3f}'.format(np.mean(dur)))
print('Duración máxima: {:.3f}'.format(np.max(dur)))
print('Duración mínima: {:.3f}'.format(np.min(dur)))

#%% Un sonograma on scipy
cual = 0;
sf,data = wavfile.read(audioPath+'/'+audioFiles[cual])

f, t, son = sg.spectrogram(data, sf,window=('gaussian',20),nperseg=0.008*sf)
plt.pcolormesh(t, f, son)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.colorbar()
plt.show()

#%% Un sonograma con matplotlib

cual = 0;
sf,data = wavfile.read(audioPath+'/'+audioFiles[cual])

ventana = sg.get_window(('gaussian',200),0.008*sf)
plt.specgram(data,Fs=sf,window=ventana,NFFT=len(ventana))

