# -*- coding: utf-8 -*-
"""
Visualizo class activation map (CAM)

"""
#importo el modelo preentrenado en la base de datos gigante
from keras.applications.vgg16 import VGG16
model = VGG16(weights='imagenet')

from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
from keras import backend as K
import matplotlib.pyplot as plt


#cargo y proceso la imagen
img_path = '/home/marcos/Pictures/elefante.png' #la imagen
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

#%% Paso la imagen por el modelo

african_elephant_output = model.output[:, 386]
last_conv_layer = model.get_layer('block5_conv3')

#pido las predicciones del modelo (tiene un softmax de 1000 categorías)
preds = model.predict(x)
print('Predicted:', decode_predictions(preds, top=3)[0])

ind = np.argmax(preds[0]) #indice en el cual está 'elefante africano' en la lista de categorias

african_elephant_output = model.output[:, ind] #elefente africano en el vector de predicción
last_conv_layer = model.get_layer('block5_conv3') #última capa convolucional

#gradiente de la clase que quiero mirar respecto de la activación
grads = K.gradients(african_elephant_output, last_conv_layer.output)[0]
#la última capa tiene 512 canales, promedio la activación en cada canal --> 512 valores
pooled_grads = K.mean(grads, axis=(0, 1, 2))
#para poder leer las cosas como np.array
iterate = K.function([model.input],
                     [pooled_grads, last_conv_layer.output[0]])

#recupero las cosas como np.array
pooled_grads_value, conv_layer_output_value = iterate([x])
#pero cada cativación con su importancia en la clase
for i in range(512): #!!! 512 es la cantidad de filtros en la ultima capa
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
#el heatmap es el promedio de la activación de cada canal 
#es de 14x14 porque es el tamaño de los filtros en la última capa
heatmap = np.mean(conv_layer_output_value, axis=-1)

#normalizo a [0,1], tirando lo <0 para matar ruido
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
plt.matshow(heatmap)

#%% Superpongo la iamgen con le heatmap

import cv2

img = cv2.imread(img_path) #vuelvo a cargar la imagen original (sin el prepross.)
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0])) #reescalo el heatmap al tamaño de la imagen
heatmap = np.uint8(255 * heatmap) #lo paso a 8bits
heatmap_RGB = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET) #le aplico el colormap jet, transformándolo en RGB
factor = 0.4 #peso del heatmap en la imagen final
superimposed_img = heatmap_RGB * factor + img * (1-factor) #sobrepongo la imagen al heatmap con peso 0.4
#guardo la imagen subrepuesta en nuevo lugar:
new_img_path = img_path[:-4] + '_cam.jpg'
cv2.imwrite(new_img_path, superimposed_img)