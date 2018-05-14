import numpy as np
import scipy

NUM_BALLENAS = 50

def generaPoblacionInicial(inf,sup,dimension,nBallenas=NUM_BALLENAS):
    poblacion_ballenas = np.array([])
    for i in range(nBallenas):
        poblacion_ballenas = np.append(poblacion_ballenas,np.random.uniform(inf,sup,dimension))
    return poblacion_ballenas.reshape(nBallenas,dimension)

def Ballena(f_obj,inf,sup,dimension,max_iter,nBallenas=NUM_BALLENAS):
    '''
    @brief Devuelve un vector de tamaño dimension que contiene la solución para la función func_objetivo
    @param f_obj Función objetivo que se pretende optimizar.
    @param inf Límite inferior para cada valor del vector de soluciones.
    @param sup Límite superior para cada valor del vector de soluciones.
    @param dimension Dimensionalidad de la función.
    @param min_max Valor booleano que indica si se minimiza o maximiza la función.
    '''

    #Inicializo la posición y score del líder
    lider_pos = np.zeros(dimension)
    lider_score = float('inf')

    #Inicializa la posición de las ballenas
    posiciones = generaPoblacionInicial(inf,sup,dimension)

    #Contador de iteraciones
    t=0

    #Valor real a
    a = 2

    #Fitness de cada ballena (inicialmente todo a ceros)
    fitness = np.zeros(nBallenas)

    #Bucle principal
    while t<max_iter:
        for i in range(len(posiciones)):
            #Devuelve a las ballenas que se han ido fuera del dominio al mismo
            posiciones[i][posiciones[i]>sup] = sup
            posiciones[i][posiciones[i]<inf] = inf

            #Calcula el fitness de cada ballena
            fitness[i] = f_obj(posiciones[i])

            #Cambia al lider si la ballena es mejor
            if fitness[i]<lider_score:
                lider_score = fitness[i]
                lider_pos = posiciones[i]

        #a se decrementa de forma lineal desde 2 hasta 0 en función de las iteraciones
        a = 2-t*(2.0/max_iter)

        #a2 se decrementa desde -1 a -2 de forma lineal
        a2 = -1+t*(-1.0/max_iter)

        for i in range(len(posiciones)):
            #Numeros aleatorios entre 0 y 1
            r1 = np.random.uniform(0,1)
            r2 = np.random.uniform(0,1)

            #Para calcular la siguiente posición de cada ballena
            A = 2*a*r1-a
            C = 2*r2

            #Parámetros de la espiral logarítmica
            b=1
            l=(a2-1)*np.random.uniform(0,1)+1

            #Número aleatorio para decidir si el movimiento es lineal o espiral
            p = np.random.uniform(0,1)

            for j in range(dimension):
                if p<0.5:
                    #Si la norma es mayor que 1 entonces hacemos una aproximación lineal a una solución aleatoria.
                    if np.absolute(A)>=1:
                        rand_lider_index = np.random.randint(0,nBallenas)
                        X_rand = posiciones[rand_lider_index]
                        D_X_rand = np.absolute(C*X_rand[j]-posiciones[i][j])
                        posiciones[i][j] = X_rand[j]-A*D_X_rand
                    #Si la norma es menor que 1 hacemos una aproximación lineal a la mejor solución
                    else:
                        D_lider = np.absolute(C*lider_pos[j]-posiciones[i][j])
                        posiciones[i][j] = lider_pos[j]-A*D_lider
                else:
                    #Hacemos la aproximación con la espiral logarítmica a la mejor solución
                    distancia_a_lider = np.absolute(lider_pos[j]-posiciones[i][j])
                    posiciones[i][j] = distancia_a_lider*np.exp(b*l)*np.cos(l*2*scipy.pi)+lider_pos[j]

        t+=1

    #Rehace los fitness y actualiza el lider
    for i in range(len(posiciones)):
        #Devuelve a las ballenas que se han ido fuera del dominio al mismo
        posiciones[i][posiciones[i]>sup] = sup
        posiciones[i][posiciones[i]<inf] = inf

        #Calcula el fitness de cada ballena
        fitness[i] = f_obj(posiciones[i])

        #Cambia al lider si la ballena es mejor
        if fitness[i]<lider_score:
            lider_score = fitness[i]
            lider_pos = posiciones[i]


    return lider_pos,lider_score
