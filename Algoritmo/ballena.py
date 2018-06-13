import numpy as np
import localsearch
import cma

NUM_BALLENAS = 50
TOLERANCIA = 0.01

PI=3.141592653589793

#Devuelve todas las funciones de la ballena implementadas
def getFuncionesBallena():
    return [Ballena1,Ballena2,Ballena3,Ballena4,Ballena5]

#Semilla aleatoria
def setSeed(seed):
    np.random.seed(seed)

def generaPoblacionInicial(inf,sup,dimension,nBallenas=NUM_BALLENAS):
    poblacion_ballenas = np.array([])
    for i in range(nBallenas):
        #poblacion_ballenas = np.append(poblacion_ballenas,np.zeros(dimension))
        poblacion_ballenas = np.append(poblacion_ballenas,np.random.uniform(inf,sup,dimension))
    return poblacion_ballenas.reshape(nBallenas,dimension)

def evolucionDiferencial(f_obj,poblacion,inf,sup,fitness):
    nBallenas = len(poblacion)
    n = np.array([])
    #Genero los vectores de ruido
    for i in range(nBallenas):
        sample = np.arange(nBallenas)
        np.random.shuffle(sample)
        sample = sample[:3]
        n = np.append(n,poblacion[sample[0]] + 0.5*(poblacion[sample[1]]-poblacion[sample[2]]))
    n = n.reshape(nBallenas,len(poblacion[0]))

    #Genero la recombinacion
    t = np.array([])
    for i in range(nBallenas):
        if np.random.uniform(0,1)<(0.1/0.9):
            t = np.append(t,n[i])
        else:
            t = np.append(t,poblacion[i])
    t = t.reshape(nBallenas,len(poblacion[0]))

    #Tomo como población recombinada los elementos según el fitness
    poblacion_recombinada = np.array([])
    fitness_recombinada = np.zeros(nBallenas)
    for i in range(nBallenas):
        fitness_t = f_obj(t[i])
        if fitness_t<fitness[i]:
            poblacion_recombinada = np.append(poblacion_recombinada,t[i])
            fitness_recombinada[i] = fitness_t
        else:
            poblacion_recombinada = np.append(poblacion_recombinada,poblacion[i])
            fitness_recombinada[i] = fitness[i]
    poblacion_recombinada = poblacion_recombinada.reshape(nBallenas,len(poblacion[0]))
    return poblacion_recombinada,fitness_recombinada

def tomaPeores(fitness,porcentaje,nBallenas):
    '''
    @param porcentaje Número entre 0 y 1 para tomar el tanto por 1 peor de la población.
    '''

    indices_ordenados = np.argsort(fitness)
    indices_ordenados = indices_ordenados[::-1]
    return indices_ordenados[:nBallenas]

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
                    posiciones[i][j] = distancia_a_lider*np.exp(b*l)*np.cos(l*2*PI)+lider_pos[j]

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
                        posiciones[i][j] = distancia_a_aleatoria*np.exp(b*l)*np.cos(l*2*PI)+X_rand[j]
                    else:
                        #Hacemos la aproximación con la espiral logarítmica a la mejor solución
                        distancia_a_lider = np.absolute(lider_pos[j]-posiciones[i][j])
                        posiciones[i][j] = distancia_a_lider*np.exp(b*l)*np.cos(l*2*PI)+lider_pos[j]

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

    #Inicializa la BL
    bl = localsearch.SolisWets(f_obj,np.array([inf,sup]),dimension)
    opciones = bl.getInitParameters(2)

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
                posiciones[s],fitness[s] = bl.improve(posiciones[s],fitness[s],1000,opciones)

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

    lider_pos,lider_score = bl.improve(lider_pos,lider_score,1000,opciones)

    return lider_pos,lider_score


################################################################################
## Descripción: Después de la búsqueda local he metido un esquema evolutivo   ##
## como el de differential evolution.                                         ##
################################################################################

def Ballena4(f_obj,inf,sup,dimension,nBallenas=NUM_BALLENAS):
    '''
    @brief Devuelve un vector de tamaño dimension que contiene la solución para la función func_objetivo
    @param f_obj Función objetivo que se pretende optimizar.
    @param inf Límite inferior para cada valor del vector de soluciones.
    @param sup Límite superior para cada valor del vector de soluciones.
    @param dimension Dimensionalidad de la función.
    @param min_max Valor booleano que indica si se minimiza o maximiza la función.
    '''

    #Inicializa la BL
    bl = localsearch.SolisWets(f_obj,np.array([inf,sup]),dimension)
    opciones = bl.getInitParameters(2)

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
    max_iter = (0.9*max_evals)//nBallenas

    #Valor real a
    a = 2

    #Fitness de cada ballena (inicialmente todo a infinito)
    fitness = np.ones(nBallenas)*float('inf')

    #Bucle principal
    while evaluaciones<0.9*max_evals:

        #Cada 1000 iteraciones hago una búsqueda local al 20% de la población
        if t%1000==0 and t!=0:
            #Tomamos las posiciones a las que hacemos la búsqueda local de forma aleatoria
            sample = np.arange(nBallenas)
            np.random.shuffle(sample)
            slice = int(nBallenas*0.25)
            sample = sample[:slice]
            for s in sample:
                evaluaciones+=1000
                posiciones[s],fitness[s] = bl.improve(posiciones[s],fitness[s],0.05*max_evals,opciones)

        #Cada 10.000 iteraciones hago un esquema de evolucion diferencial.
        if t%10000==0 and t!=0:
            posiciones,fitness = evolucionDiferencial(f_obj,posiciones,inf,sup,fitness)


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

            if p<0.9:
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
                sample = tomaPeores(fitness,0.5,nBallenas)
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

    lider_pos,lider_score = bl.improve(lider_pos,lider_score,max_evals-evaluaciones,opciones)

    return lider_pos,lider_score

################################################################################
## Descripción: Cambio de la búsqueda local por CMAES.                        ##
################################################################################

def Ballena5(f_obj,inf,sup,dimension,nBallenas=NUM_BALLENAS):
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
    max_iter = (0.9*max_evals)//nBallenas

    #Valor real a
    a = 2

    #Fitness de cada ballena (inicialmente todo a infinito)
    fitness = np.ones(nBallenas)*float('inf')

    #Bucle principal
    while evaluaciones<max_evals:

        #Cada 1000 iteraciones hago una búsqueda local al 20% de la población
        if t%100==0 and t!=0:
            #Tomamos las posiciones a las que hacemos la búsqueda local de forma aleatoria
            sample = np.arange(nBallenas)
            np.random.shuffle(sample)
            slice = int(nBallenas*0.25)
            sample = sample[:slice]
            for s in sample:
                es = cma.CMAEvolutionStrategy(posiciones[s],np.std(posiciones[s]),{'seed': 123456789,'verb_disp':0})
                es.optimize(f_obj,verb_disp=0)
                posiciones[s] = es.result[0]
                fitness[s] = es.result[1]
                evaluaciones+=es.result[3]

        #Cada 10.000 iteraciones hago un esquema de evolucion diferencial.
        if t%50==0 and t!=0:
            posiciones,fitness = evolucionDiferencial(f_obj,posiciones,inf,sup,fitness)


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

            if p<0.9:
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
                sample = tomaPeores(fitness,0.5,nBallenas)
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

    es = cma.CMAEvolutionStrategy(lider_pos,np.std(lider_pos),{'seed': 123456789,'verb_disp':0})
    es.optimize(f_obj,verb_disp=0)
    lider_pos = es.result[0]
    lider_score = es.result[1]
    evaluaciones+=es.result[3]

    return lider_pos,lider_score
