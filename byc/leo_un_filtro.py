import os
import numpy as np
from keras.models import load_model
from keras import backend as K

#esto es para poder correr remotamente
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils import bitificar8, filcol, new_name

#%% cambio el directorio y cargo modelo

ubicacion = 'modelos'
modelo = load_model(os.path.join(ubicacion, 'Benteveos_Chingolos.h6'))
#modelo.summary() #recuerdo qué tenía

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

layer, index = 2, 3
nombre = modelo.layers[layer].name
este_filtro = generate_pattern(nombre, index, modelo, iteraciones=100)
este_filtro = np.squeeze(este_filtro)

#print(este_filtro.shape)

plt.imshow(este_filtro) #cmap=plt.get_cmap('Greys'))
plt.axis('off')
nombre = new_name(os.path.join('filtros','un_filtro_capa{}_ind{}.jpg'.format(layer, index)))
plt.savefig(nombre, dpi=50, facecolor='w', edgecolor='w',
            orientation='portrait', papertype=None, format=None,
            transparent=False, bbox_inches=None, pad_inches=0.1,
            frameon=None)
#    plt.show()
plt.close()