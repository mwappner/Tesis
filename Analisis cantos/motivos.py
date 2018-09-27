# -*- coding: utf-8 -*-
"""
Estudio motivos

"""

from scipy.io import wavfile
import numpy as np
import matplotlib
matplotlib.use("Qt4Agg") # This program works with Qt only
import pylab as pl
from PIL import Image
import os
#from skimage import filters #para otsu

motivoPath = '/home/marcos/Documents/Codigos/Analisis cantos/Motivos'
sonoPath = os.path.join(motivoPath,'Sonogramas')
sonoFiles = os.listdir(sonoPath)
sonoFiles.sort()

nombres = open(os.path.join(motivoPath,'orden_nombres.txt'),'r')
contenido_nombres = nombres.readlines()
nombres.close()
contenido_nombres = [nombre[:-1] for nombre in contenido_nombres] #le saco el '\n'

duraciones = open(os.path.join(motivoPath,'duraciones.txt'),'r')
contenido_duraciones = duraciones.readlines()
duraciones.close()

dur = {k:float(v) for k,v in zip(contenido_nombres,contenido_duraciones)}
#hacer un diccionario con nombre --> duración

    


#%% Duración de cada archivo:

#calculo duracion de un archivo
def duracion(path,file):
    sf,data = wavfile.read(path + '/' + file)
    return len(data)/sf

#devuelve el ancho de la ventana actual
def Ancho(axes):
    xa = axes.get_xlim()
    return xa[1]-xa[0]

#seteo el nuevo ancho como 
def NuevoAncho(axes):
    ancho = Ancho(axes)
    xla = axes.set_xlabel('Ancho: {:.3f}s'.format(ancho),fontsize=18,fontweight='bold')
    return xla
    
#cambio el título para agregale un string
def NuevoTitulo(ultima_entrada,titulo_viejo):
    ax1.set_title(titulo_viejo + ultima_entrada)
    
#recupero los digitos al final de un string
def final(st):
    out = ''
    while st[-1].isdigit():
        out = st[-1] + out
        st = st[:-1]
    return out, st

#reemplazo 'ff' por frecuencias
def IncertoFrecuencias(the_string):
    #por default escribe -fmin-fmax- donde decía -ff-
    #si la quiero al revés, uso una i después de la categoría
    #la exponencial va invertida por default
    ya = ax1.get_ylim()
    if the_string[1]=='i' or the_string[0]=='e':
        ya = ya[::-1] #invierto el orden
        the_string = the_string.replace('i','') #saco la 'i' porque ya invertí
    #obtengo las frecuencias
    frecuencias = '-{:.3f}-{:.3f}-'.format(ya[0],ya[1])
    the_string = the_string.replace('-ff-',frecuencias)
    return the_string

print('Duración media: {:.3f}'.format(np.mean(list(dur.values()))))
print('Duración máxima: {:.3f}'.format(np.max(list(dur.values()))))
print('Duración mínima: {:.3f}'.format(np.min(list(dur.values()))))

#%% Importo y trabajo sonogramas


escala = 2500 #cantidad de pixeles del sonograma por segundo
cual = 17 #qué figura miro
nombre_actual = sonoFiles[cual][:-4]

#cargo imagen, transformo a escala de grises
sono1 = Image.open(sonoPath + '/' + sonoFiles[cual]).convert('L')
#recorto la parte que me interesa (dentro de los ejes)
sono1 = sono1.crop((360, 238, 6845, 1860))
#reescalo la imagen para que el tamaño refleje la duración
#convierto a np.array
alto = sono1.size[1]
ancho = int(dur[sonoFiles[cual]] * escala)
A = np.array(sono1.resize((ancho,alto),Image.LANCZOS))
A = 255-A #la escala está al revés (255 es poca potencia)


#Plots:
fig, ax1 = pl.subplots()
fig.set_size_inches([18.39,  9.27])

ax1.imshow(A,cmap='Greys',extent=[0, dur[sonoFiles[cual]],0, 0.1])
#ax1.plot(tiempo[:-1],diffThreshCu*0.5,'g--')

titulo = '{}: {}, duración:{:.1f}s'.format(cual,nombre_actual,dur[sonoFiles[cual]])
ax1.set_title(titulo)
xla = NuevoAncho(ax1)

#a1 = plt.subplot(2,1,2,sharex=a0)
#plt.plot(tiempo,intens)
#plt.plot(tiempo,cuadrado,'r')
#plt.plot(tiempo,threshCu,'g--')

### control panel ###
from PyQt4 import QtGui
from PyQt4.QtCore import Qt

escritura.close() #la cierro por si las moscas no la cerré antes
escritura = open(os.path.join(motivoPath,'Secuencias',nombre_actual) + '.txt','w')

def Escribo(the_string):
    NuevoTitulo(the_string,titulo)
    print(the_string)
    escritura.write(the_string)

def update():
    te = textbox.text() #texto actualizado
    fig.canvas.draw_idle()
    if te[-1]==' ':
        valor = te[:-1] #sin el espacio
        
        #reemplazo ff por frecuencias
        valor = IncertoFrecuencias(valor)
            
        #si escribo 'close ', cierra el archivo de escritura
        if valor=='close':
            escritura.close()
            NuevoTitulo('\nCERRADO',titulo)
            print('Cerrado')
        
        #si no me gustó, printeo 'no'        
        elif valor=='no':
            Escribo(valor)
            NuevoTitulo('',titulo)
            
        #chequeo que empiece correctamente: letra, guión, número, o sólo letra, guión.
        elif valor[0].isalpha() and valor[1] == '-' and (len(valor)==2 or valor[2].isdigit()):
            #si tengo termina en '-', escribo el input y el ancho de ventana
            if valor[-1]=='-':
                a = Ancho(ax1)
                s = '\n{}{:.3f}'.format(valor,a)
                Escribo(s)
                
            #si tengo termina en número, formateo el numero como duracion
            elif valor[-1].isdigit():
                el_numero , comienzo = final(valor)
                s = '\n{}0.{}'.format(comienzo,el_numero)
                Escribo(s)
                
            else:
                print('No escribí: ',valor)
    
        else: 
            print('No escribí: ',valor)


root = fig.canvas.manager.window
panel = QtGui.QWidget()
hbox = QtGui.QHBoxLayout(panel)
textbox = QtGui.QLineEdit(parent = panel)
textbox.textChanged.connect(update) #cada vez que cambio el texto, llama update()
hbox.addWidget(textbox)
panel.setLayout(hbox)

dock = QtGui.QDockWidget("Completar secuencia:", root)
root.addDockWidget(Qt.BottomDockWidgetArea, dock)
dock.setWidget(panel)

#cuando ahgo zoom, lo escribo en el xlabel
ax1.callbacks.connect('xlim_changed',NuevoAncho)

######################

pl.show()