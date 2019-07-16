
import os
import numpy as np
from keras.models import load_model
from keras import backend as K

#esto es para poder correr remotamente
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils import bitificar8, Grid, new_name, make_dirs_noreplace

#%% cambio el directorio y cargo modelo

ubicacion = 'modelos/byc_peque_pad'
nombre_modelo = 'byc.h6'
modelo = load_model(os.path.join(ubicacion, nombre_modelo))
modelo.summary() #recuerdo qué tenía

# Carpeta donde guarda las imagenes de los filtros
# save_dir = make_dirs_noreplace(os.path.join(
#         'filtros', os.path.splitext(nombre_modelo)[0]))
save_dir = ubicacion

#cre una func. que arma la imagen visualizada
def generate_pattern(layer_name, filter_index, modelo, iteraciones=40):
    
    layer_output = modelo.get_layer(layer_name).output #la capa
    loss = K.mean(layer_output[:, :, :, filter_index]) #algo que maximiza con la capa
    grads = K.gradients(loss, modelo.input)[0] #gradiente de esto respecto a un input
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5) #normalizo y sumo un poquito para no dividir por cero
    iterate = K.function([modelo.input], [loss, grads]) #función a iterar
    input_img_data = np.random.random((1, *modelo.input_shape[1:])) * 20 + 128. #la imagen arranca siendo gris ruidoso
    
    step = 1. #tamaño del avance del gradiente
    for _ in range(iteraciones): #itero [iteraciones] veces
        loss_value, grads_value = iterate([input_img_data]) 
        input_img_data += grads_value * step #nuevo input corrigiendo con el gradiente

    img = input_img_data[0]
    return bitificar8(img,0.1)

#%% Grafico filtros
    
# Nomres de las capas convolutivas
nombres = [l.name for l in modelo.layers if 'conv' in l.name]

# Itero sobre todos los filtros de las capas convolutivas
for capa_ind, nombre in enumerate(nombres):
    print('Corriendo capa '+ nombre)

    # Cantidad de filtros en esta capa
    cant_filtros = modelo.layers[capa_ind].output_shape[3] #cant de filtros
    # Una grilla donde meto las imagenes de cada filtro
    g = Grid(cant_filtros, fill_with=0, trasponer=True) 

    # Proceso los filtros y los meto en la grilla
    for filtro_ind in range(cant_filtros):
        channel_filter = generate_pattern(nombre, filtro_ind, modelo)
        channel_filter = np.squeeze(channel_filter)

        # channel_filter es la imagen en cuestión
        g.insert_image(channel_filter)

    # plt.imshow(g.grid)
    g.show()
    nombre = new_name(os.path.join(save_dir, 'filtros_{}.jpg'.format(nombre)))
    plt.savefig(nombre)
    plt.close()
