import os
import linecache

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import contenidos, find_numbers

ubi = 'modelos'
modelos = contenidos(ubi)

def extraigo(file):
    nr_lineas = (76, 81)
    lineas = []
    for l in range(*nr_lineas):
        lineas.append(linecache.getline(file, l))
#    print(lineas)
    
    resultados = {}
    # Matriz de confusión
    resultados['mc'] = np.array(
            [find_numbers(lineas.pop(0)),
             find_numbers(lineas.pop(0))])
    
    # Métricas
    for l in lineas:
        l = l[:-1] #corto el \n
        for x in l.split(' , '):
            k, v = x.split(' = ')
            resultados[k] = float(v)
    
    return resultados
    
        
def get_resultados(ubicacion):
    try:
        r = contenidos(ubicacion, filter_ext='.txt')[0]
    except IndexError:
        print(ubicacion, ' no contenìa resultados')
        return
    return extraigo(r)
    
#%% Extraigo valores
todos = {}
for k in contenidos(ubi):
    res = get_resultados(k)
    if res is not None:
        res['name'] = k
        todos[os.path.basename(k).replace('byc_pad_', '')] = res
        
#%% Grafico accuracy

plt.plot(list(todos.keys()), [v['acc'] for v in todos.values()], 'o')
plt.xticks(rotation='vertical')

# Pad margins so that markers don't get clipped by the axes
plt.margins(0.2)
# Tweak spacing to prevent clipping of tick-labels
plt.subplots_adjust(bottom=0.2)
plt.grid()

#%% Todos 2 (para DataFrames)

todos_df = []

for k in contenidos(ubi):
    res = get_resultados(k)
    if res is not None:
        res['nombre'] = os.path.basename(k).replace('byc_pad_', '')
        res['ubicacion'] = k
        todos_df.append(res)

data = pd.DataFrame(todos_df)
data.set_index('name', inplace=True)