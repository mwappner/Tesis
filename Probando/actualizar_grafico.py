# -*- coding: utf-8 -*-
"""
Prueba de actualización de info de grafico
"""

import matplotlib.pyplot as plt

def Ancho(axes):
    xa = axes.get_xlim()
    return xa[1]-xa[0]
#
# Some toy data
x_seq = [x / 100.0 for x in range(1, 100)]
y_seq = [x**2 for x in x_seq]

#
# Scatter plot
fig, ax = plt.subplots(1, 1)
ax.scatter(x_seq, y_seq)
ancho = Ancho(ax)
ax.set_xlabel('{:.2f}'.format(ancho))

#
# Declare and register callbacks
def on_xlims_change(axes):
    print("updated xlims: ", ax.get_xlim())

def on_ylims_change(axes):
    print("updated ylims: ", ax.get_ylim())
    
def NuevoAncho(axes):
    ancho = Ancho(axes)
    axes.set_xlabel('{:.2f}'.format(ancho))

#ax.callbacks.connect('xlim_changed', on_xlims_change)
#ax.callbacks.connect('ylim_changed', on_ylims_change)

ax.callbacks.connect('xlim_changed',NuevoAncho)

#
# Show
plt.show()
