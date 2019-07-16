# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 07:51:34 2018

@author: gabo
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 16:34:44 2018

@author: gabrielmindlin
"""


from keras import layers
from keras import models
from keras import optimizers
from keras import regularizers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.preprocessing import image
import numpy as np
#from keras.preprocessing import image
#import numpy as np


train_dir= '/home/gabo/Desktop/Chingolos/train'
validation_dir= '/home/gabo/Desktop/Chingolos/validation'

model=models.Sequential()
model.add(layers.Conv2D(4,(3,3),kernel_regularizer=regularizers.l2(0.001),activation='relu',input_shape=(300,200,1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(8, (3, 3),kernel_regularizer=regularizers.l2(0.001), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(8, (3, 3),kernel_regularizer=regularizers.l2(0.001), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(16, (3, 3),kernel_regularizer=regularizers.l2(0.001), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(3, activation='softmax'))

#model.compile(loss='categorical_crossentropy',optimizer=optimizers.RMSprop(lr=1e-4),metrics=['acc'])
model.compile(loss='categorical_crossentropy',optimizer=optimizers.RMSprop(lr=1e-4),metrics=['acc'])

train_datagen=ImageDataGenerator(
        rescale=1./255,)
#        rotation_range=40,
#        width_shift_range=0.8,
#        height_shift_range=0.8,)
#        shear_range=0.2,
#        zoom_range=0.4,)
#        horizontal_flip=True,)
validation_datagen=ImageDataGenerator(1./255)

#aca vienen los generadores, que son objetos que funcan como iteradores. De paso, normalizamos
train_generator=train_datagen.flow_from_directory(
        train_dir,
        target_size=(300,200),
        color_mode='grayscale',
        batch_size=10,
        class_mode='categorical')
validation_generator=validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(300,200),
        color_mode='grayscale',
        batch_size=10,
        class_mode='categorical')

# El metodo flow_from_directory va a generar batches, y seran data batches y label batches.
# Los labels salen del path

history=model.fit_generator(
        train_generator,
        steps_per_epoch=200,
        epochs=15,
        # epochs=15
        validation_data=validation_generator,
        validation_steps=200) 
        # validation_steps=133

model.save('Chingolos.h6')

acc=history.history['acc']
val_acc=history.history['val_acc']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(1,len(acc)+1)
plt.plot(epochs,acc,'bo',label='training acc')
plt.plot(epochs,val_acc,'b',label='validation acc')
plt.title('training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs,loss,'bo',label='training loss')
plt.plot(epochs,val_loss,'b',label='validation loss')
plt.title('training and validation loss')
plt.legend()

plt.show()

img_path='/home/gabo/Desktop/Chingolos/sonograma_chingolo_3.jpeg'
img=image.load_img(img_path,target_size=(300,200),grayscale=True)
#img.reshape([-1,300, 200,1])
x=image.img_to_array(img)
x=np.expand_dims(x,axis=0)
preds=model.predict(x)
print(preds)
#
#img_path='/home/gabo/Desktop/Benteveos_Chingolos_14/2.jpeg'
#img=image.load_img(img_path,target_size=(300,200))
#x=image.img_to_array(img)
#x=np.expand_dims(x,axis=0)
#preds=model.predict(x)
#print(preds)

