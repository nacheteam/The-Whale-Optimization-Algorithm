import numpy as np
import scipy

NUM_BALLENAS = 50
TOLERANCIA = 0.01

#Devuelve todas las funciones de la ballena implementadas
def getFuncionesBallena():
    return [Ballena1,Ballena2,Ballena3]

#Semilla aleatoria
def setSeed(seed):
    np.random.seed(seed)

def generaPoblacionInicial(inf,sup,dimension,nBallenas=NUM_BALLENAS):
    poblacion_ballenas = np.array([])
    for i in range(nBallenas):
        #poblacion_ballenas = np.append(poblacion_ballenas,np.zeros(dimension))
        poblacion_ballenas = np.append(poblacion_ballenas,np.random.uniform(inf,sup,dimension))
    return poblacion_ballenas.reshape(nBallenas,dimension)

def busquedaLocal(f_obj,inf,sup,dimension,solucion,max_evals):
    evals = 1
    score = f_obj(solucion)
    sol_local = solucion
    while evals<max_evals:
        vecino = sol_local+np.random.uniform(-0.1,0.1,dimension)
        vecino[vecino<inf] = inf
        vecino[vecino>sup] = sup
        score_vecino = f_obj(vecino)
        if score>score_vecino:
            score = score_vecino
            sol_local = vecino
        evals+=1
    return sol_local,score

################################################################################
## Descripción: Función de la ballena primigenia. Es una traducción del       ##
## algoritmo implementado por los autores de la metaheurística de Matlab      ##
## a Python.                                                                  ##
################################################################################
def Ballena1(f_obj,inf,sup,dimension,nBallenas=NUM_BALLENAS):
    '''
    @brief Devuelve un vector de tamaño dimension que contiene la solución para la función func_objetivo
    @param f_obj Función objetivo que se pretende optimizar.
    @param inf Límite inferior para cada valor del vector de soluciones.
    @param sup Límite superior para cada valor del vector de soluciones.
    @param dimension Dimensionalidad de la función.
    @param min_max Valor booleano que indica si se minimiza o maximiza la función.
    '''

    #Inicializo el número de evaluaciones
    max_evals = 10000*dimension
    evaluaciones=0

    #Inicializo la posición y score del líder
    lider_pos = np.random.uniform(inf,sup,dimension)
    lider_score = float('inf')

    #Inicializa la posición de las ballenas
    posiciones = generaPoblacionInicial(inf,sup,dimension)

    #Contador de iteraciones
    t=0
    max_iter = max_evals//nBallenas

    #Valor real a
    a = 2

    #Fitness de cada ballena (inicialmente todo a ceros)
    fitness = np.zeros(nBallenas)

    #Bucle principal
    while evaluaciones<max_evals:
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

        #Sumo nBallenas evaluaciones después de recalcular el fitness
        evaluaciones+=nBallenas

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


################################################################################
## Descripción: Antes el movimiento a una solución aleatoria sólo se hacía    ##
## de forma lineal, ahora también se hace de forma espiral cuando el          ##
## algoritmo está en una fase inicial.
################################################################################

def Ballena2(f_obj,inf,sup,dimension,nBallenas=NUM_BALLENAS):
    '''
    @brief Devuelve un vector de tamaño dimension que contiene la solución para la función func_objetivo
    @param f_obj Función objetivo que se pretende optimizar.
    @param inf Límite inferior para cada valor del vector de soluciones.
    @param sup Límite superior para cada valor del vector de soluciones.
    @param dimension Dimensionalidad de la función.
    @param min_max Valor booleano que indica si se minimiza o maximiza la función.
    '''

    #Inicializo el número de evaluaciones
    max_evals = 10000*dimension
    evaluaciones=0

    #Inicializo la posición y score del líder
    lider_pos = np.random.uniform(inf,sup,dimension)
    lider_score = float('inf')

    #Inicializa la posición de las ballenas
    posiciones = generaPoblacionInicial(inf,sup,dimension)

    #Contador de iteraciones
    t=0
    max_iter = max_evals//nBallenas

    #Valor real a
    a = 2

    #Fitness de cada ballena (inicialmente todo a ceros)
    fitness = np.zeros(nBallenas)

    #Bucle principal
    while evaluaciones<max_evals:
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

        #Sumo nBallenas evaluaciones después de recalcular el fitness
        evaluaciones+=nBallenas

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
                    if np.absolute(A)>=1:
                        #Hacemos una aproximación en espiral a una solución aleatoria cuando estamos en una fase inicial del algoritmo.
                        rand_lider_index = np.random.randint(0,nBallenas)
                        X_rand = posiciones[rand_lider_index]
                        distancia_a_aleatoria = np.absolute(X_rand[j]-posiciones[i][j])
                        posiciones[i][j] = distancia_a_aleatoria*np.exp(b*l)*np.cos(l*2*scipy.pi)+X_rand[j]
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



################################################################################
## Descripción: He cambiado el movimiento en espiral por una mutación         ##
## aleatoria del 20% de la población. Así mismo he metido que se hace la      ##
## búsqueda local a la mitad de la población cada 1000 iteraciones y al final ##
################################################################################

def Ballena3(f_obj,inf,sup,dimension,nBallenas=NUM_BALLENAS):
    '''
    @brief Devuelve un vector de tamaño dimension que contiene la solución para la función func_objetivo
    @param f_obj Función objetivo que se pretende optimizar.
    @param inf Límite inferior para cada valor del vector de soluciones.
    @param sup Límite superior para cada valor del vector de soluciones.
    @param dimension Dimensionalidad de la función.
    @param min_max Valor booleano que indica si se minimiza o maximiza la función.
    '''

    #Inicializo el número de evaluaciones
    max_evals = 10000*dimension
    evaluaciones=0

    #Inicializo la posición y score del líder
    lider_pos = np.random.uniform(inf,sup,dimension)
    lider_score = float('inf')

    #Inicializa la posición de las ballenas
    posiciones = generaPoblacionInicial(inf,sup,dimension)

    #Contador de iteraciones
    t=0
    max_iter = (max_evals-1000)//nBallenas

    #Valor real a
    a = 2

    #Fitness de cada ballena (inicialmente todo a ceros)
    fitness = np.zeros(nBallenas)

    #Bucle principal
    while evaluaciones<max_evals-1000:

        if t%1000==0 and t!=0:
            sample = np.arange(nBallenas)
            np.random.shuffle(sample)
            slice = int(nBallenas*0.5)
            sample = sample[:slice]
            for s in sample:
                evaluaciones+=1000
                posiciones[s],fitness[s] = busquedaLocal(f_obj,inf,sup,dimension,posiciones[s],1000)

        for i in range(len(posiciones)):
            #Devuelve a las ballenas que se han ido fuera del dominio al mismo
            posiciones[i][posiciones[i]>sup] = sup
            posiciones[i][posiciones[i]<inf] = inf

            #Calcula el fitness de cada ballena
            fitness[i] = f_obj(posiciones[i])

            #Cambia al lider si la ballena es mejor
            if fitness[i]<lider_score:
                lider_score = np.copy(fitness[i])
                lider_pos = np.copy(posiciones[i])

        #Sumo nBallenas evaluaciones después de recalcular el fitness
        evaluaciones+=nBallenas

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

            if p<0.5:
                #Si la norma es mayor que 1 entonces hacemos una aproximación lineal a una solución aleatoria.
                if np.absolute(A)>=1:
                    rand_lider_index = np.random.randint(0,nBallenas)
                    X_rand = posiciones[rand_lider_index]
                    D_X_rand = np.absolute(C*X_rand-posiciones[i])
                    posiciones[i] = X_rand-A*D_X_rand
                #Si la norma es menor que 1 hacemos una aproximación lineal a la mejor solución
                else:
                    D_lider = np.absolute(C*lider_pos-posiciones[i])
                    posiciones[i] = lider_pos-A*D_lider
            else:
                sample = np.arange(nBallenas)
                np.random.shuffle(sample)
                slice = int(nBallenas*0.2)
                sample = sample[:slice]
                for s in sample:
                    posiciones[s] = np.random.uniform(inf,sup,dimension)

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
            lider_score = np.copy(fitness[i])
            lider_pos = np.copy(posiciones[i])

    lider_pos,lider_score = busquedaLocal(f_obj,inf,sup,dimension,lider_pos,1000)

    return lider_pos,lider_score
