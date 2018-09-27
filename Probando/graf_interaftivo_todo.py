# -*- coding: utf-8 -*-
"""
Textbox interactivo con ancho
"""

import numpy as np
import matplotlib
matplotlib.use("Qt4Agg") # This program works with Qt only
import pylab as pl

def Ancho(axes):
    xa = axes.get_xlim()
    return xa[1]-xa[0]

fig, ax1 = pl.subplots()

t = np.linspace(0, 10, 200)

line, = ax1.plot(t, np.sin(t))
ax1.set_xlabel('{:.2f}'.format(Ancho(ax1)))


### control panel ###
from PyQt4 import QtGui
from PyQt4 import QtCore
from PyQt4.QtCore import Qt

escritura = open('datos.txt','w')

def update():
    te = textbox.text() #texto actualizado
    fig.canvas.draw_idle()
    if te[-1]==' ':
        valor = te[:-1] #sin el espacio
#        print(valor)
        #si escribo 'close ', cierra el archivo de escritura
        if valor=='close':
            escritura.close()
            print('Cerrado')
            
        #si tengo '[letra] ', escribo la letra y el ancho de ventana
        elif len(valor)==1:
            a = Ancho(ax1)
            s = '{} {:.2f}\n'.format(valor,a)
            print(s)
            escritura.write(s)
            
        #si no es sólo una letra, chequeo que sea letra+numero y nada más
        elif (valor[1:].isdigit() and valor[0].isalpha()):
            s = '{} 0.{}\n'.format(valor[0],valor[1:])
            print(s)
            escritura.write(s)
        else: 
            print('No escribí: ',valor)

#escribo el ancho de la figura
def NuevoAncho(axes):
    ancho = Ancho(axes)
    axes.set_xlabel('{:.2f}'.format(ancho))
            
root = fig.canvas.manager.window
panel = QtGui.QWidget()
hbox = QtGui.QHBoxLayout(panel)
textbox = QtGui.QLineEdit(parent = panel)
textbox.textChanged.connect(update) #cada vez que cambio el texto, llama update()
hbox.addWidget(textbox)
panel.setLayout(hbox)

dock = QtGui.QDockWidget("control", root)
root.addDockWidget(Qt.BottomDockWidgetArea, dock)
dock.setWidget(panel)

ax1.callbacks.connect('xlim_changed',NuevoAncho)

######################

pl.show()

#%%

escritura.close()
