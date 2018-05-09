import numpy as np

def Ballena(f_obj,inf,sup,dimension,min_max):
    '''
    @brief Devuelve un vector de tamaño dimension que contiene la solución para la función func_objetivo
    @param f_obj Función objetivo que se pretende optimizar.
    @param inf Límite inferior para cada valor del vector de soluciones.
    @param sup Límite superior para cada valor del vector de soluciones.
    @param dimension Dimensionalidad de la función.
    @param min_max Valor booleano que indica si se minimiza o maximiza la función.
    '''
