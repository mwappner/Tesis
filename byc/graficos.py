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
        print(ubicacion, ' no contenía resultados')
        return
    return extraigo(r)
        
#%% Extraigo valores

todos_df = []

for k in contenidos(ubi):
    res = get_resultados(k)
    if res is not None:
        res['nombre'] = os.path.basename(k).replace('byc_pad_', '')
        res['ubicacion'] = k
        todos_df.append(res)

data = pd.DataFrame(todos_df)
data.set_index('nombre', inplace=True)


#%% Grafico accuracy
fig, (acc, prec, rec) = plt.subplots(3,1, sharex=True)

acc.plot(data.index, data['acc'], 'o', color='C2')
acc.grid()
acc.set_ylabel('Accuracy')

prec.plot(data.index, data['precB'],'d', label='benteveos')
prec.plot(data.index, data['precC'],'*', label='chingolos')
prec.grid()
prec.legend()
prec.set_ylabel('Precicion')

rec.plot(data.index, data['recB'],'d', label='benteveos')
rec.plot(data.index, data['recC'],'*', label='chingolos')
rec.grid()
rec.legend()
rec.set_ylabel('Recall')

rec.set_xlim([-0.2,19.1])
plt.xticks(rotation=90)

fig.tight_layout()
fig.subplots_adjust(hspace=0)


# Pad margins so that markers don't get clipped by the axes
plt.margins(0.2)
# Tweak spacing to prevent clipping of tick-labels
plt.subplots_adjust(bottom=0.2)
