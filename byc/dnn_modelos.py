from keras import layers, models, regularizers

def peque(im_size):
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
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(2, activation='softmax'))

    return model


def peque_conectada(im_size):
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


def peque_densa(im_size):
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
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(2, activation='softmax'))

    return model


def media(im_size):
    model = models.Sequential()
    model.add(layers.Conv2D(4, kernel_size=7, strides=2, input_shape=(*im_size, 1),
              kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.MaxPooling2D(4)) # Será mucho?
    model.add(layers.Conv2D(8, kernel_size=5, strides=1, 
              kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.MaxPooling2D(4))
    model.add(layers.Conv2D(8, kernel_size=3, strides=1, 
              kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.MaxPooling2D(2))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(2, activation='softmax'))

    return model


def grande(im_size):
    model = models.Sequential()
    model.add(layers.Conv2D(8, kernel_size=7, strides=1, input_shape=(*im_size, 1),
              kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.MaxPooling2D(4)) # Será mucho?
    model.add(layers.Conv2D(16, kernel_size=5, strides=1, 
              kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.MaxPooling2D(4))
    model.add(layers.Conv2D(32, kernel_size=3, strides=1, 
              kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.MaxPooling2D(2))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(2, activation='softmax'))

    return model


def grande_shallow(im_size):
    model = models.Sequential()
    model.add(layers.Conv2D(8, kernel_size=7, strides=1, input_shape=(*im_size, 1),
              kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.MaxPooling2D(4)) # Será mucho?
    model.add(layers.Conv2D(16, kernel_size=5, strides=1, 
              kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.MaxPooling2D(4))
    model.add(layers.Conv2D(32, kernel_size=3, strides=1, 
              kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.MaxPooling2D(2))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(2, activation='softmax'))
    

def profunda(im_size):
    model = models.Sequential()
    model.add(layers.Conv2D(4, kernel_size=5, strides=1, input_shape=(*im_size, 1),
              kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.MaxPooling2D(2)) # Será mucho?
    model.add(layers.Conv2D(4, kernel_size=5, strides=1, 
              kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.MaxPooling2D(2))
    model.add(layers.Conv2D(8, kernel_size=3, strides=1, 
              kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.MaxPooling2D(2))
    model.add(layers.Conv2D(16, kernel_size=3, strides=1, 
              kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.MaxPooling2D(2))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(2, activation='softmax'))

    return model


def mas_profunda(im_size):
    model = models.Sequential()
    model.add(layers.Conv2D(4, kernel_size=5, strides=1, input_shape=(*im_size, 1),
              kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.MaxPooling2D(2)) # Será mucho?
    model.add(layers.Conv2D(4, kernel_size=5, strides=1, 
              kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.MaxPooling2D(2))
    model.add(layers.Conv2D(8, kernel_size=3, strides=1, 
              kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.MaxPooling2D(2))
    model.add(layers.Conv2D(16, kernel_size=3, strides=1, 
              kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.MaxPooling2D(2))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(2, activation='softmax'))

    return model


def asimetrica(im_size):
    model = models.Sequential()
    model.add(layers.Conv2D(4, kernel_size=(5,3), strides=2, input_shape=(*im_size, 1),
              kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.MaxPooling2D(4)) # Será mucho?
    model.add(layers.Conv2D(4, kernel_size=(5,3), strides=1, 
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
    peque = peque,
    peque_conectada = peque_conectada,
    peque_densa = peque_densa,
    media = media,
    grande = grande,
    grande_shallow = grande_shallow,
    profunda = profunda,
    mas_profunda = mas_profunda,
    asimetrica = asimetrica,
    )
