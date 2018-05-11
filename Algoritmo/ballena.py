import numpy as np

NUM_BALLENAS = 50

def generaPoblacionInicial(inf,sup,dimension,nBallenas=NUM_BALLENAS):
    poblacion_ballenas = np.array([])
    for i in range(nBallenas):
        poblacion_ballenas = np.append(poblacion_ballenas,np.random.uniform(inf,sup,dimension))
    return poblacion_ballenas.reshape(nBallenas,dimension)

def Ballena(f_obj,inf,sup,dimension,nBallenas=NUM_BALLENAS):
    '''
    @brief Devuelve un vector de tamaño dimension que contiene la solución para la función func_objetivo
    @param f_obj Función objetivo que se pretende optimizar.
    @param inf Límite inferior para cada valor del vector de soluciones.
    @param sup Límite superior para cada valor del vector de soluciones.
    @param dimension Dimensionalidad de la función.
    @param min_max Valor booleano que indica si se minimiza o maximiza la función.
    '''
