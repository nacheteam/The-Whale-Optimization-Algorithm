import numpy as np
import scipy

################################################################################
##                             CONSTANTES GLOBALES                            ##
################################################################################

MIN_BOUND = -100
MAX_BOUND = 100
D=10
num_funciones = 16
matrices = []
optimos = []

#Para la función 5
e = 2.718281

#Para la función 6
a=0.5
b=3
kmax=20

################################################################################
##                        OBTENCION DE LAS FUNCIONES                          ##
################################################################################

def getFunciones():
    funciones = []
    funciones.append(f1)
    funciones.append(f2)
    funciones.append(f3)
    funciones.append(f4)
    funciones.append(f5)
    funciones.append(f6)
    funciones.append(f7)
    funciones.append(f8)
    funciones.append(f9)
    funciones.append(f10)
    funciones.append(f11)
    funciones.append(f12)
    funciones.append(f13)
    funciones.append(f14)
    funciones.append(f15)
    funciones.append(f16)
    return funciones

################################################################################
##                  MODIFICACIÓN DE LAS CONSTANTES GLOBALES                   ##
################################################################################

def setDimension(dim):
    global D
    D=dim

def getMinBound():
    global MIN_BOUND
    return MIN_BOUND

def getMaxBound():
    global MAX_BOUND
    return MAX_BOUND

#Hay que ejecutarla antes de utilizar ninguna función
def obtenerOptimosMatrices():
    global matrices
    global optimos

    #Lee las matrices y optimos desde 1 hasta num_funciones
    matrices = [leeMatrices(i) for i in range(1,num_funciones+1)]
    optimos = [leeOptimos(i) for i in range(1,num_funciones+1)]

################################################################################
##                           FUNCIONES AUXILIARES                             ##
################################################################################

## Auxiliar de la función 13
def auxf4(x):
    return np.sum(100*(x[:-1]*x[:-1] - x[1:]*x[1:]) + (x[:-1]-np.ones(2-1))*(x[:-1]-np.ones(2-1)))

def auxf7(x):
    return np.sum(x*x)*(1.0/4000) - np.prod(np.divide(np.cos(x),np.sqrt(np.arange(1,2+1)))) + 1

## Auxiliar de la función 14
def auxg(x,y):
    return 0.5 + ( np.float_power(np.sin(np.sqrt(x*x+y*y)),2) - 0.5 ) / ( (1+0.001*(x*x+y*y))*(1+0.001*(x*x+y*y)) )

## Auxiliar de las funciones híbridas
def leeMatrices(num_func):
    matriz = []
    with open("./matrices_optimos/M_{0}_D{1}.txt".format(num_func,D),"r") as f:
        for line in f:
            matriz.append(list(filter(None,line[:-1].split(" "))))
        for i in range(len(matriz)):
            for j in range(len(matriz[i])):
                matriz[i][j] = float(matriz[i][j])

    return np.array(matriz)

def leeOptimos(num_func):
    optimos = []
    with open("./matrices_optimos/shift_data_{0}.txt".format(num_func),"r") as f:
        optimos = list(filter(None,f.read()[:-1].split(" ")))
        for i in range(len(optimos)):
            optimos[i] = float(optimos[i])
    return np.array(optimos[:D])

################################################################################
##                         FUNCIONES BÁSICAS CEC 2014                         ##
################################################################################

#High Conditioned Elliptic Function
def Basicaf1(x,D=D):
    return np.sum(np.float_power(np.repeat(1000000,D),(range(1,D+1)-np.ones(D))/(np.repeat(D,D)-np.ones(D))) * (x*x))

#Bent Cigar Function
def Basicaf2(x,D=D):
    return np.sum(np.append(1,np.repeat(1000000,D-1))*x*x)

#Discus Function
def Basicaf3(x,D=D):
    return np.sum(np.append(1000000,np.repeat(1,D-1))*x*x)

#Rosenbrock's Function
def Basicaf4(x,D=D):
    return np.sum(100*(x[:-1]*x[:-1] - x[1:])*(x[:-1]*x[:-1] - x[1:]) + (x[:-1]-np.ones(D-1))*(x[:-1]-np.ones(D-1)))

#Ackley's Function
def Basicaf5(x,D=D):
    return -20*np.exp(-0.2*np.sqrt((1/D)*np.sum(x*x))) - np.exp((1/D)*np.sum(np.cos(2*scipy.pi*x))) + 20 + e

#Weierstrass Function
#Estoy usando 'a' directamente sin llamar a la constante por la definición de la función lambda.
def Basicaf6(x,D=D):
    x_mat = np.cos((np.repeat(0.5,(kmax+1)*D).reshape(kmax+1,D) + np.repeat(x,kmax+1).reshape(kmax+1,D))*2*scipy.pi*np.repeat(np.float_power(np.repeat(b,kmax+1),range(kmax+1)),D).reshape(kmax+1,D))
    ak_mat = np.repeat(np.fromfunction(lambda i,j: np.float_power(0.5,i*j+j),(1,kmax+1))[0],D).reshape(kmax+1,D)
    bk = np.cos(0.5*2*scipy.pi*np.fromfunction(lambda i,j: np.float_power(3,i*j+j),(1,kmax+1))[0])
    ak = np.fromfunction(lambda i,j: np.float_power(0.5,i*j+j),(1,kmax+1))[0]
    return np.sum(np.sum(x_mat*ak_mat,axis=1)) - D*np.sum(ak*bk)

#Griewank's Function
def Basicaf7(x,D=D):
    return np.sum(x*x)*(1.0/4000) - np.prod(np.divide(np.cos(x),np.sqrt(np.arange(1,D+1)))) + 1

#Rastrigin's Function
def Basicaf8(x,D=D):
    return np.sum(x*x - 10*np.cos(2*scipy.pi*x)+10)

#Modified Schwefel's Function
def Basicaf9(x,D=D):
    z = x + np.repeat(4.029687462275036e+002,D)
    res_z = np.zeros(D)
    for i in range(len(z)):
        if z[i]>500:
            res_z[i]=(500-z[i]%500)*np.sin(np.sqrt(np.absolute(500-z[i]%500)))-((z[i]-500)*(z[i]-500)/(10000*D))
        elif z[i]<-500:
            res_z[i] = (-z[i]%500 - 500)*np.sin(np.sqrt(np.absolute(-z[i]%500)-500))-((z[i]+500)*(z[i]+500)/(10000*D))
        else:
            res_z[i] = z[i]*np.sin(np.sqrt(np.absolute(z[i])))

    return 418.9829*D - np.sum(res_z)

#Katsuura Function
#Parece que si funciona pero no dan el mismo resultado la de numpy y la iterativa
def Basicaf10(x,D=D):
    '''
    res = 10.0/(D*D)
    for i in range(D):
        sum = 0
        for j in range(1,33):
            sum+=np.divide(np.absolute(np.float_power(2,j)*x[i]-np.around(np.float_power(2,j)*x[i])),np.float_power(2,j))
        res*=np.float_power(1+(i+1)*sum,(10/(np.float_power(D,12))))
    res-=10/(D*D)
    return res'''

    x_mat = np.repeat(x,32).reshape(D,32)
    j2 = np.tile(np.float_power(np.ones(32)*2,np.arange(1,33)),10).reshape(D,32)
    return (10.0/(D*D))*np.prod(np.float_power(np.ones(D)+np.arange(1,D+1)*np.sum(np.divide(j2*x_mat - np.around(j2*x_mat),j2),axis=1),np.repeat(10/(D**12),D)),axis=0) - 10.0/(D*D)

#HappyCat Function
def Basicaf11(x,D=D):
    return np.float_power(np.absolute(np.sum(x*x-D,axis=0)),(1.0/4.0)) + (0.5*np.sum(x*x,axis=0) + np.sum(x,axis=0))/D + 0.5

#HGBat Function
def Basicaf12(x,D=D):
    return np.sqrt(np.absolute(np.float_power(np.sum(x*x,axis=0),2)-np.float_power(np.sum(x,axis=0),2))) + (0.5*np.sum(x*x,axis=0) + np.sum(x,axis=0))/D + 0.5

#Expanded Griewank's plus Rosenbrock's Function
def Basicaf13(x,D=D):
    res = 0
    for i in range(D-1):
        res+=auxf7(auxf4(np.array([x[i],x[i+1]])))
    res+=auxf7(auxf4(np.array([x[-1],x[0]])))
    return res

#Expanded Scaffer's F6 Function
def Basicaf14(x,D=D):
    res = 0
    for i in range(D-1):
        res+=auxg(x[i],x[i+1])
    res+=auxg(x[-1],x[0])
    return res


################################################################################
##                         FUNCIONES OBJETIVO CEC 2014                        ##
################################################################################

#Rotated High Conditioned Elliptic Function
def f1(x):
    return Basicaf1(matrices[0].dot(x-optimos[0])) + 100

#Rotated Bent Cigar Function
def f2(x):
    return Basicaf2(matrices[1].dot(x-optimos[1])) + 200

#Rotated Discus Function
def f3(x):
    return Basicaf3(matrices[2].dot(x-optimos[2])) + 300

#Shifted and Rotated Rosenbrock's Function
def f4(x):
    return Basicaf4(matrices[3].dot( np.divide((x-optimos[3])*2.048,100)) + np.ones(D)) + 400

#Shifted and Rotated Ackley's Function
##### MAL #####
def f5(x):
    return Basicaf5(matrices[4].dot(x-optimos[4])) + 800

#Shifted and Rotated Weierstrass Function
def f6(x):
    return Basicaf6(matrices[5].dot( np.divide(0.5*(x-optimos[5]),100) )) + 600

#Shifted and Rotated Griewank's Function
##### MAL #####
def f7(x):
    return Basicaf7(matrices[6].dot( np.divide(600*(x-optimos[6]),100) )) + 700

#Shifted Rastrigin's Function
##### MAL #####
def f8(x):
    return Basicaf8(np.divide(5.12*(x-optimos[7]),100))+700

#Shifted and Rotated Rastrigin's Function
def f9(x):
    return Basicaf8(matrices[8].dot(np.divide(5.12*(x-optimos[8]),100))) + 900

#Shifted Schwefel's Function
##### MAL #####
def f10(x):
    return Basicaf9(np.divide(1000*(x-optimos[9]),100)) + 1000

#Shifted and Rotated Schwefel's Function
##### MAL #####
def f11(x):
    return Basicaf9(matrices[10].dot(np.divide(1000*(x-optimos[10]),100)))+1100

#Shifted and Rotated Katsuura Function
def f12(x):
    return Basicaf10(matrices[11].dot(np.divide(5*(x-optimos[11]),100))) + 1200

#Shifted and Rotated HappyCat Function
##### MAL #####
def f13(x):
    return Basicaf11(matrices[12].dot(np.divide(5*(x-optimos[12]),100))) + 1300

#Shifted and Rotated HGBat Function
##### MAL #####
def f14(x):
    return Basicaf12(matrices[13].dot(np.divide(5*(x-optimos[13]),100)))+1400

#Shifted and Rotated Expanded Griewank's plus Rosenbrock's Function
##### MAL #####
def f15(x):
    return Basicaf13(matrices[14].dot(np.divide(5*(x-optimos[14]),100)))+1500

#Shifted and Rotated Expanded Scaffer's F6 Function
##### MAL #####
def f16(x):
    return Basicaf14(matrices[15].dot(x-optimos[15])+np.ones(D)) + 1600
