import numpy as np
import matplotlib.pyplot as plt

CANT_SINTESIS = 10000
MAX_PUNTOS = 7
MIN_DIST = 3
TAM_IMG = 32

def chequear_distancias(antiguos, nuevo, min_dist):
    
    if not antiguos: #si no hay viejos
        return True
    
    distancias = [np.linalg.norm(x-nuevo) for x in antiguos]
    return min(distancias) > min_dist

def agregar_punto(imagen, coord):
    imagen[coord[0]-1:coord[0]+2, coord[1]] = 1
    imagen[coord[0],coord[1]-1:coord[1]+2] = 1
   
cant_puntos = np.random.randint(0, MAX_PUNTOS, CANT_SINTESIS)
imagen = np.zeros((CANT_SINTESIS, TAM_IMG, TAM_IMG))

for i, cant in enumerate(cant_puntos):
    coords = []
    while len(coords)<cant:
        nueva = np.random.randint(2, TAM_IMG-2, 2)
        if chequear_distancias(coords, nueva, MIN_DIST):
            coords.append(nueva)
            agregar_punto(imagen[i,:,:], nueva)


np.savez('circulos', imagen=imagen, cant_puntos=cant_puntos)