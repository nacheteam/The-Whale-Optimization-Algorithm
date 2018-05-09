import numpy as np

def Ballena(func_objetivo,lim_inf,lim_sup,dimension,min_max):
    '''
    @brief Devuelve un vector de tamaño dimension que contiene la solución para la función func_objetivo
    @param func_objetivo Función objetivo que se pretende optimizar.
    @param lim_inf Límite inferior para cada valor del vector de soluciones.
    @param lim_sup Límite superior para cada valor del vector de soluciones.
    @param dimension Dimensionalidad de la función.
    @param min_max Valor booleano que indica si se minimiza o maximiza la función.
    '''
