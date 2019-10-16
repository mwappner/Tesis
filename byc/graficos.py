import os
import linecache

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import contenidos, find_numbers, load_img

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
        if os.path.isdir(ubicacion):
            r = contenidos(ubicacion, filter_ext='.txt')[0]
    except (IndexError, NotADirectoryError):
        print(ubicacion, ' no contenía resultados')
        return
    return extraigo(r)
        
#%% Extraigo valores

cant_params = {}
with open('modelos/cant_params.txt', 'r') as f:
    for line in f.readlines():
        modelo, cant = line.split()
        cant_params[modelo] = int(cant.rstrip())
cant_params['VGG16'] = np.nan
        
todos_df = []

for k in contenidos(ubi):
#    print(k)
    if os.path.isdir(k):
        res = get_resultados(k)
#        print('Funcionó')
    else:
        continue
    if res is not None:
        n = os.path.basename(k).replace('byc_pad_', '') #nombre recortado
        res['nombre'] = n
        res['ubicacion'] = k
        key = n[:-2] if n[-1].isdigit() and n[-2]=='_' else n
        res['nparams'] = cant_params[key]
        
        todos_df.append(res)

data = pd.DataFrame(todos_df)
data.set_index('nombre', inplace=True)

#%% Selecciono las que quiero graficar
seleccionados = [
        'peque_conectada_2', 
        'asimetrica_2',
        'grande_shallow_4',
#        'mas_profunda',
        'peque_densa_2',
        'profunda_2',        
        ]



#%% Grafico accuracy

pdata = data.loc[seleccionados]
pdata.sort_values('nparams', inplace=True)

nombres = 'pequeña', 'asimétrica',  'pequeña\ndensa','grande','grande\nprofunda'
#xax_vals = [x for _, x in zip(seleccionados, 'ABCDEFGHIJKLMN')] #para A, B, C, D
xax_vals = [x for _, x in zip(seleccionados,nombres)]

#pdata = data.copy()
pdata.sort_values('nparams', inplace=True)
#xax_vals = list(pdata.index)

fig, (acc, prec, rec, cant) = plt.subplots(4,1, sharex=True)
fig.set_size_inches([5, 6 ])

acc.plot(xax_vals, pdata['acc'], 'o', color='C0')
acc.grid() 
acc.set_ylabel('Exactitud')
acc.set_ylim(.85, 1.03)

prec.plot(xax_vals, pdata['precB'],'d', label='benteveos', color='C1')
prec.plot(xax_vals, pdata['precC'],'*', label='chingolos', color='C2')
prec.grid()
#prec.legend()
prec.set_ylabel('Precisión')
prec.set_ylim(.8, 1.03)

rec.plot(xax_vals, pdata['recB'],'d', label='benteveos', color='C1')
rec.plot(xax_vals, pdata['recC'],'*', label='chingolos', color='C2')
rec.grid()
rec.set_ylabel('Exhaust.')
rec.set_ylim(.8, 1.03)
#rec.legend(loc='lower left', framealpha=1, edgecolor='k', bbox_to_anchor=(0, -.5))

cant.bar(xax_vals, pdata['nparams'],color='C0')
cant.grid()
cant.set_ylabel('Cant. params.')
cant.set_xlabel('Modelo')

#Para que aparezca la leyenda de los otros plots
cant.plot(0,0,'d', label='benteveos', color='C1')
cant.plot(0,0,'*', label='chingolos', color='C2')
cant.legend(loc='upper left', framealpha=1, edgecolor='k', bbox_to_anchor=(0,1.35))

#cant.set_xlim([-0.4,4.4])
cant.set_yscale('log')
plt.xticks(rotation=60)


fig.tight_layout()
fig.subplots_adjust(hspace=0.07)

## Pad margins so that markers don't get clipped by the axes
plt.margins(0.2)
## Tweak spacing to prevent clipping of tick-labels
plt.subplots_adjust(bottom=0.3)
fig.savefig('plots/cinco_modelos_muy_grande_colores.png', dpi=600)

#%% ============ GRAFICO SONOGRAMAS ============

bent = 'sintetizados/dnn/pad/test/benteveo'
ching = 'sintetizados/dnn/pad/test/chingolo'

#bent_ims = contenidos(bent)
i0 = np.array(load_img(contenidos(bent)[0], color_mode='grayscale'))
i0 = np.zeros(i0.shape)

for im in contenidos(bent):
    i0 += np.array(load_img(im, color_mode='grayscale'))
    
i0 /= i0.max()    
plt.imshow(i0, cmap='viridis')

    




