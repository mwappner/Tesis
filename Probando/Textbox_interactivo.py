# -*- coding: utf-8 -*-
"""
Textbox interactivo
"""

import numpy as np
import matplotlib
matplotlib.use("Qt4Agg") # This program works with Qt only
import pylab as pl
fig, ax1 = pl.subplots()

t = np.linspace(0, 10, 200)

line, = ax1.plot(t, np.sin(t))

### control panel ###
from PyQt4 import QtGui
from PyQt4 import QtCore
from PyQt4.QtCore import Qt

def Ancho(axes):
    xa = axes.get_xlim()
    return xa[1]-xa[0]

escritura = open('datos.txt','w')

def update():
    categoria = textbox.text()
    fig.canvas.draw_idle()
    if categoria[-1]==' ':
        a = Ancho(ax1)
        s = '{} {:.2f\n}'.format(categoria[:-1],a)
        print(s)
        escritura.write(s)

            
root = fig.canvas.manager.window
panel = QtGui.QWidget()
hbox = QtGui.QHBoxLayout(panel)
textbox = QtGui.QLineEdit(parent = panel)
textbox.textChanged.connect(update)
hbox.addWidget(textbox)
panel.setLayout(hbox)

dock = QtGui.QDockWidget("control", root)
root.addDockWidget(Qt.BottomDockWidgetArea, dock)
dock.setWidget(panel)
######################

pl.show()

#%%

escritura.close()
