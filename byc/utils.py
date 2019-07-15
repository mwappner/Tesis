import numpy as np
import os
import time
import re

from scipy.signal import spectrogram
from scipy.io import wavfile
from scipy.ndimage import gaussian_filter
from skimage import filters


#para correr remotamente
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def bitificar8(im,desv=1):
    '''Devuelve la imagen pasada a 8 bits. Puede reescalar la desviación.'''
    #normalizo la imagen            
    im -= im.mean()
    im /= (im.std() + 1e-5) #por si tengo cero
    im *= desv
    #la llevo a 8 bits
    im *= 64
    im += 128
    im = np.clip(im, 0, 255).astype('uint8') #corta todo por fuera de la escala
    return im

def new_name(name, newseparator='_'):
    '''Returns a name of a unique file or directory so as to not overwrite.
    
    If proposed name existed, will return name + newseparator + number.
     
    Parameters:
    -----------
        name : str (path)
            proposed file or directory name influding file extension
        nweseparator : str
            separator between original name and index that gives unique name
            
    Returns:
    --------
        name : str
            unique namefile using input 'name' as template
    '''
    
    #if file is a directory, extension will be empty
    base, extension = os.path.splitext(name)
    i = 2
    while os.path.exists(name):
        name = base + newseparator + str(i) + extension
        i += 1
        
    return name

def make_dirs_noreplace(dirs_paths):
    try:
        os.makedirs(dirs_paths)
    except FileExistsError:
        print('While creating ', dirs_paths, 'found it already exists.')

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

class contenidos(list):

    def __init__(self, carpeta, full_path=True, natsort=True):
        '''Si full_path=True, los elementos de la lista serán los contenidos de la carpeta
        apendeados con el nombre de la carpeta en sí. Si no, serán sólo los contenidos, en
        cuyo caso habrá algunas funconalidads no disponibles.'''
        self.carpeta = carpeta
        self.full_path = full_path
        self.update()
        if natsort:
            self.natural_sort()

    def update(self):
        if self.full_path:
            super().__init__((os.path.join(self.carpeta, f) for f in os.listdir(self.carpeta)))
        else:
            super().__init__(os.listdir(self.carpeta))

    def natural_sort(self):
        convert = lambda text: int(text) if text.isdigit() else text.lower() 
        self.sort(key=lambda key: [convert(c) for c in re.split('([0-9]+)', key)])

    def filtered_ext(self, extension):
        '''Crea una nueva lista de las cosas con la extensión correspondiente.'''
        return [elem for elem in self if elem.endswith(extension)]

    def filter_ext(self, extension):
        '''Elimina de la lista todo lo que no tenga la extensión correspondiente'''
        super().__init__(self.filtered_ext(extension))

    def files(self):
        '''Devuelve nueva lista de sólo los elementos que son archivos.'''
        return [elem for elem in self if os.path.isfile(elem)]
        
    def keep_fies(self):
        '''Elimina de la lista todo lo que no sean archivos.'''
        super().__init__(self.files())

    def directories(self):
        '''Devuelve nueva lista de sólo los elementos que son carpetas.'''
        return [elem for elem in self if os.path.isdir(elem)]
    
    def keep_dirs(self):
        '''Elimina de la lista todo lo que no sean carpetas.'''
        super().__init__(self.directories())

class Testimado:

    def __init__(self, cant_total):
        self.cant_total = cant_total
        self.inicio = time.time()

    def restante(self, indice):
        return round((self.cant_total / (indice+1) - 1) * self.transcurrido())

    def transcurrido(self):
        return time.time() - self.inicio
    
    def horas_minutos(self, i):
         horas, rem = divmod(self.restante(i), 3600)
         minutos = rem//60
         return horas, minutos
         
    def horas_minutos_segundos(self, i):
         horas, rem = divmod(self.restante(i), 3600)
         minutos, segundos= divmod(rem, 60)
         return (horas, minutos, segundos)
     
    def time_str(self, i, include_times = 'HM'):
        '''Devuelve un string con el tiempo restante formateado según se indica
        en include_times.
        j: días
        H: horas
        M: minutos
        S: segundos'''
        format_tring = ':'.join('%{}'.format(s) for s in include_times)
        return time.strftime(format_tring, time.gmtime(self.restante(i)))
    
    def print_remaining(self, i, *a, **kw):
        print('ETA: {}'.format(self.time_str(i, *a, **kw)))

class FiltroSonograma:
    
    def __init__(self, archivo, target_duration=None, limites_frec=(10, 8000), *args, **kwargs):
        self.nombre = archivo
        self.lims = limites_frec
        self.target_dur = target_duration
        
        self.fs, self.sonido = wavfile.read(archivo)
        self._hago_sonograma(sigma=.15, *args, **kwargs)

    def _hago_sonograma(self, dur_seg=0.012, overlap=.9, sigma=.25, 
                       gauss_filt={'sigma':0.1}):
        '''Calcula un sonograma con ventana gaussiana. Overlap y sigma son 
        proporcionales al tamaño de la ventana. Tamaño de la ventana dado en 
        segundos. Devuelve un sonograma en escala logarítmica.
        Por defecto aplica un filtro gaussiano de orden 1 a la salida usando 
        sigma=0.1. Para no aplicarlo, fijar gauss_filt a diccionario vacío o a
        False.'''
        if not 0<=overlap<=1 or not 0<=sigma<=1:
            raise ValueError('overlap y sigma deben estar entre 0 y 1.')
        
        nperseg = int(dur_seg*self.fs)
        f, t, sxx = spectrogram(self.sonido, fs=self.fs,
                           window=('gaussian',nperseg*sigma),
                           nperseg=nperseg,
                           noverlap=int(nperseg*overlap),
                           scaling='spectrum')
        
        sxx = sxx[np.logical_and(f>self.lims[0], f<self.lims[1]), :]
        self.frecuencias = f[np.logical_and(f>self.lims[0], f<self.lims[1])]
        self.tiempos = t
        self.target_dur = self.target_dur or t[-1]
        
        if gauss_filt:
            self.sono = np.log10(gaussian_filter(sxx + 1e-6, sigma=1))
        else:
            self.sono = np.log10(sxx + 1e-6)
        
        
    def bitificar8(self, desv=1, ceros=True):
        '''Devuelve la imagen pasada a 8 bits. Puede reescalar la desviación.
        La variable ceros define si se deben tener en cuenta los ceros en la 
        imagen a la hora de calcular media y desviación.'''
        #normalizo la imagen a [-1, 1]
        if self.sono.dtype == np.float32:
            im = self.sono
        else:
            im = self.sono.astype(np.float32)
        #Decido si contar o no los ceros en el cálculo d la media
        if ceros:        
            im -= im.mean()
        else:
            im -= im[im>0].mean()
            
        im /= (im.std() + 1e-5) #por si tengo cero
        im *= desv
        #la llevo a 8 bits
        im *= 64
        im += 128
        im = np.clip(im, 0, 255).astype('uint8') #corta todo por fuera de la escala
        self.sono = im
    
    
    def normalizar(self, bits=True):
        '''Devuelve una copia del sonograma rescalado para ocupat todo el rango
        dinámico del formato correspondiente. Si bits=True, devuelve una imagen
        en 8 bits. Si no, la escala será [0,1].'''
        
        im = self.sono
        
        im -= im.min()
        im /= im.max() + 1e-10 #por si tengo cero
        if not bits:
            return im
        im *= 255
        self.sono = im.astype('uint8')
    
    
    def thresholdear(self):
        '''Filtra una imagen utilziando el criterio de otsu.'''
        im = self.sono
        self.sono = np.where(im>filters.threshold_otsu(im), im, 0)
    
    
    def rango_dinamico(self, valor, bits=True):
        '''Escala la imagen de forma que todos los valores que estén un porcentaje 
        <valor> debajo del máximo pasan a cero y el máximo pasa a ser 1 o 255,
        dependiendo de si bits=True o False, respecivamente.'''
        imagen = self.sono

        if not 0<=valor<=1:
            raise ValueError('valor debe ser entre 0 y 1')
        self.normalizar(bits=False) #valores ahora en [0,1]
        imagen[imagen<valor] = valor
        self.normalizar(bits)
    
    
    def cut_or_extend(self, centered=False):
        '''Si la duración actual del sonido es más grande que la del sonido 
        objetivo, recorta el final. Si es más chica, rellena con ceros.'''
        if self.tiempos[-1] > self.target_dur:
            self.sono = self.sono[:, self.tiempos<self.target_dur]
            
        elif self.tiempos[-1] < self.target_dur:
            dt = self.tiempos[1] - self.tiempos[0]
            cant_faltante = int((self.target_dur - self.tiempos[-1])/dt)
            
            if centered:
                antes = int(np.floor(cant_faltante/2))
                despues = int(np.ceil(cant_faltante/2))
                patron = ((0,0),(antes, despues))
            else:
                patron = ((0,0),(0,cant_faltante))
            
            # lleno con ceros
            self.sono = np.pad(self.sono, pad_width=patron, 
                               mode='constant', constant_values=0)

        #redefino el vector de tiempos para graficar 
        self.tiempos = np.linspace(self.tiempos[0], self.target_dur, self.sono.shape[1])
            
    
    def plotear(self, im=None, ax=None, log=False, labels=False):
        '''Formatea una imagen con título y ejes, y la plotea.
            im: imagen a graficar. Default: self.sono
            ax: eje donde graficar. Default: nuevo eje
            log: [True|False] tomar logaritmo de los datos antes de graficar.
            labels: [True|False] Decide si poner o no labels.'''
            
        im = im or self.sono #si no le di una imagen, uso el guardado
        
        if ax is None:
            fig, ax = plt.subplots()
        if log:
            im = np.log10(im)
        ax.pcolormesh(self.tiempos, self.frecuencias/1000, im,
                      rasterized=True, cmap=plt.get_cmap('Greys'))
        if labels:
            plt.xlabel('tiempo [s]')
            plt.ylabel('frecuencia [Hz]')
            plt.title(self.archivo, fontsize=15)


    def guardar(self, nombre=None, ubicacion=None, extension='jpg'):
        '''Guarda el sonograma en 'ubicacion/nombre.extension'. Si no se le da nombre
        o ubicación, utiliza el del archivo anterior. Notar que nunca reemplazará dicho 
        archivo.'''
        
        #Chequeo valores y fijo defaults
        extensiones_validas = ('jpg', 'jpeg', 'png')
        if extension.lower() not in extensiones_validas:
            raise ValueError('Extension debe ser una de {}'.format(extensiones_validas))
        
        u, n = os.path.split(self.nombre)
        ubicacion = ubicacion or u
        nombre = nombre or os.path.splitext(n)[0]
        
        #Ploteo
        fig, ax = plt.subplots()
        ax.pcolormesh(self.tiempos, self.frecuencias,
                      self.sono,
                      rasterized=True,
                      cmap=plt.get_cmap('Greys'))
        ax.set_ylim(self.lims)
        ax.axis('off')
        fig.subplots_adjust(bottom = 0, top = 1, left = 0, right = 1) #para que no tenga bordes blancos
        
        #Guardo
        nombre = new_name(os.path.join(ubicacion, '.'.join((nombre,extension))))
        fig.savefig(nombre, dpi=100)
        plt.close(fig)
        
        print('Saved: {}'.format(nombre))

        return nombre

class Grid:
    '''Una clase para crear y llenar una grilla con imagenes.'''
       
    def __init__(self, cant, fill_with=np.nan, trasponer=False, bordes=True):

        self.cant = cant #cantidad de imagenes
        self.trasponer = trasponer #por default, la grilla es más ancha que alta
        self.bordes = bordes #si debe o no haber un margen entre figus.
        self.shape = self._cant_to_mat() #tamaño de la matriz de imagenes

        self.grid = None #la grilla a llenar con imagenes
        #self.im_shape = None #tamaño de la imagen
        self.ind = 0 #por qué imagen voy?
        self.fill_with = fill_with #con qué lleno la grilla vacía

    @property
    def im_shape(self):
        return self._im_shape_real
    @im_shape.setter
    def im_shape(self, value):
        self._im_shape_real = value
        self._imRGB = len(value)==3 #if image is RGB
        
        if self.bordes:
            self._im_shape_bordes = (value[0] + 1, value[1] + 1)
        else:
            self._im_shape_bordes = self.im_shape

    def _cant_to_mat(self):
        '''Dimensiones de la cuadrícula más pequeña y más cuadrada
        posible que puede albergar [self.cant] cosas.'''
        col = int(np.ceil(np.sqrt(self.cant)))
        row = int(round(np.sqrt(self.cant)))
        if self.trasponer:
            return col, row
        else:
            return row, col
        
    def _filcol(self):
        '''Pasa de índice lineal a matricial.'''
        fil = self.ind // self.shape[1]
        col = self.ind % self.shape[1]
        return int(fil), int(col)

    def _create_grid(self):
        shape = (self._im_shape_bordes[0] * self.shape[0], 
                 self._im_shape_bordes[1] * self.shape[1])
        if self.bordes:
            shape = shape[0] + 1, shape[1] + 1
        if self._imRGB:
            shape = *shape, 3
            
        self.grid = np.full(shape, self.fill_with)

    def insert_image(self, im):
        '''Agrego una imagen a la grilla.'''
        #inicializo la grilla
        if self.grid is None:
            self.im_shape = im.shape
            self._create_grid()

        #la lleno
        col, row = self._filcol()
        #sumo el booleano de bordes para que cierre bien la cuenta
        if self._imRGB:
            self.grid[col * self._im_shape_bordes[0] + int(self.bordes) : 
                        (col + 1) * self._im_shape_bordes[0],
                    row * self._im_shape_bordes[1] + int(self.bordes) : 
                        (row + 1) * self._im_shape_bordes[1], :]= im
        else:        
            self.grid[col * self._im_shape_bordes[0] + int(self.bordes) : 
                        (col + 1) * self._im_shape_bordes[0],
                    row * self._im_shape_bordes[1] + int(self.bordes) : 
                        (row + 1) * self._im_shape_bordes[1]]= im
        
        #avanzo el contador apra la siguiente imagen
        self.ind += 1
        
    def show(self, **kw):
        if self.grid is None:
            raise ValueError('No se insertaron imagenes aún.')
        plt.imshow(self.grid, cmap=kw.pop('cmap', 'viridis'), **kw)

#Grid Testing
# cant = 33
# shape = (21,25)
# g = Grid(cant, trasponer=False, bordes=True, fill=100)
# for i in range(cant):
#     g.insert_image(np.ones(shape)*i)

# plt.matshow(g.grid)
# plt.grid()

# #%%

# cant = 17
# shape = (11,9)
# g = Grid(cant, trasponer=False, bordes=True, fill=np.nan)
# colores = [(0,0,0), (1,0,0), (0,1,0), (0,0,1), (0,1,1), (1,0,1), (1,1,0), 
#            (1,1,1), (.5,.5,.5), (1,.5,0), (1,0,.5), (.5,1,0), (.5,0,1),
#            (0,.5,1), (0,1,.5), (.5,0,0), (0,.5,0), (0,0,.5)]
# imagenes = []
# for c in colores:
#     liso = np.ones((*shape,3))
#     for i in range(3):
#         liso[:,:,i] *= c[i]
#     imagenes.append(liso)

# for i in range(cant):
#     g.insert_image(imagenes[i])

# plt.imshow(g.grid)
# plt.grid()

