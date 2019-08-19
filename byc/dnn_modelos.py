from keras import models

def peque_pad():
    model = models.Sequential()
    model.add(layers.Conv2D(4, kernel_size=5, strides=2, input_shape=(*im_size, 1),
              kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.MaxPooling2D(4)) # Será mucho?
    model.add(layers.Conv2D(4, kernel_size=3, strides=1, 
              kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.MaxPooling2D(2))
    model.add(layers.Conv2D(8, kernel_size=3, strides=1, 
              kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.MaxPooling2D(2))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(2, activation='softmax'))

    return model


def pad():
    model = models.Sequential()
    model.add(layers.Conv2D(4, kernel_size=5, strides=2, input_shape=(*im_size, 1),
              kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.MaxPooling2D(4)) # Será mucho?
    model.add(layers.Conv2D(4, kernel_size=3, strides=1, 
              kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.MaxPooling2D(2))
    model.add(layers.Conv2D(8, kernel_size=3, strides=1, 
              kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.MaxPooling2D(2))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(2, activation='softmax'))

    return model


def grande_pad():
    model = models.Sequential()
    model.add(layers.Conv2D(4, kernel_size=5, strides=2, input_shape=(*im_size, 1),
              kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.MaxPooling2D(4)) # Será mucho?
    model.add(layers.Conv2D(4, kernel_size=3, strides=1, 
              kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.MaxPooling2D(2))
    model.add(layers.Conv2D(8, kernel_size=3, strides=1, 
              kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.MaxPooling2D(2))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(2, activation='softmax'))

    return model


def profunda_pad():
    model = models.Sequential()
    model.add(layers.Conv2D(4, kernel_size=5, strides=2, input_shape=(*im_size, 1),
              kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.MaxPooling2D(4)) # Será mucho?
    model.add(layers.Conv2D(4, kernel_size=3, strides=1, 
              kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.MaxPooling2D(2))
    model.add(layers.Conv2D(8, kernel_size=3, strides=1, 
              kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.MaxPooling2D(2))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(2, activation='softmax'))

    return model


def peque_stretch():
    model = models.Sequential()
    model.add(layers.Conv2D(4, kernel_size=5, strides=2, input_shape=(*im_size, 1),
              kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.MaxPooling2D(4)) # Será mucho?
    model.add(layers.Conv2D(4, kernel_size=3, strides=1, 
              kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.MaxPooling2D(2))
    model.add(layers.Conv2D(8, kernel_size=3, strides=1, 
              kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.MaxPooling2D(2))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(2, activation='softmax'))

    return model


def stretch():
    model = models.Sequential()
    model.add(layers.Conv2D(4, kernel_size=5, strides=2, input_shape=(*im_size, 1),
              kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.MaxPooling2D(4)) # Será mucho?
    model.add(layers.Conv2D(4, kernel_size=3, strides=1, 
              kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.MaxPooling2D(2))
    model.add(layers.Conv2D(8, kernel_size=3, strides=1, 
              kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.MaxPooling2D(2))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(2, activation='softmax'))

    return model


switcher = dict(
    peque_pad=peque_pad,
    pad=pad,
    grande_pad=grande_pad,
    profunda_pad=profunda_pad,
    peque_stretch=peque_stretch,
    stretch=stretch,
    )
