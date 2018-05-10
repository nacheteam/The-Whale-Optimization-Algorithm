import numpy as np
import scipy

################################################################################
##                             CONSTANTES GLOBALES                            ##
################################################################################

MIN_BOUND = -100
MAX_BOUND = 100
D=10

#Para la función 5
e = 2.718281

#Para la función 6
a=0.5
b=3
kmax=20

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
def leeMatriz(num_func):
    

################################################################################
##                         FUNCIONES OBJETIVO CEC 2014                        ##
################################################################################

#High Conditioned Elliptic Function
def f1(x):
    return np.sum(np.float_power(np.repeat(1000000,D),(range(1,D+1)-np.ones(D))/(np.repeat(D,D)-np.ones(D))) * (x*x))

#Bent Cigar Function
def f2(x):
    return np.sum(np.append(1,np.repeat(1000000,D-1))*x*x)

#Discus Function
def f3(x):
    return np.sum(np.append(1000000,np.repeat(1,D-1))*x*x)

#Rosenbrock's Function
def f4(x):
    return np.sum(100*(x[:-1]*x[:-1] - x[1:]*x[1:]) + (x[:-1]-np.ones(D-1))*(x[:-1]-np.ones(D-1)))

#Ackley's Function
def f5(x):
    return -20*np.exp(-0.2*np.sqrt((1/D)*np.sum(x*x))) - np.exp((1/D)*np.sum(np.cos(2*scipy.pi*x))) + 20 + e

#Weierstrass Function
#Estoy usando 'a' directamente sin llamar a la constante por la definición de la función lambda.
def f6(x):
    x_mat = np.cos((np.repeat(0.5,(kmax+1)*D).reshape(kmax+1,D) + np.repeat(x,kmax+1).reshape(kmax+1,D))*2*scipy.pi*np.repeat(np.float_power(np.repeat(b,kmax+1),range(kmax+1)),D).reshape(kmax+1,D))
    ak_mat = np.repeat(np.fromfunction(lambda i,j: np.float_power(0.5,i*j+j),(1,kmax+1))[0],D).reshape(kmax+1,D)
    bk = np.cos(0.5*2*scipy.pi*np.fromfunction(lambda i,j: np.float_power(3,i*j+j),(1,kmax+1))[0])
    ak = np.fromfunction(lambda i,j: np.float_power(0.5,i*j+j),(1,kmax+1))[0]
    return np.sum(np.sum(x_mat*ak_mat,axis=1)) - D*np.sum(ak*bk)

#Griewank's Function
def f7(x):
    return np.sum(x*x)*(1.0/4000) - np.prod(np.divide(np.cos(x),np.sqrt(np.arange(1,D+1)))) + 1

#Rastrigin's Function
def f8(x):
    return np.sum(x*x - 10*np.cos(2*scipy.pi*x)+10)

#Modified Schwefel's Function
def f9(x):
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
#Seguramente mal porque con np.ones(10) y sus múltiplos y np.arange(10) me da cero.
def f10(x):
    x_mat = np.repeat(x,32).reshape(D,32)
    j2 = np.tile(np.float_power(np.ones(32)*2,np.arange(1,33)),10).reshape(D,32)
    return (10.0/(D*D))*np.prod(np.float_power(np.ones(D)+np.arange(1,D+1)*np.sum(np.divide(j2*x_mat - np.around(j2*x_mat),j2),axis=1),np.repeat(10/(D**12),D)),axis=0) - 10.0/(D*D)

#HappyCat Function
def f11(x):
    return np.float_power(np.absolute(np.sum(x*x-D,axis=0)),(1.0/4.0)) + (0.5*np.sum(x*x,axis=0) + np.sum(x,axis=0))/D + 0.5

#HGBat Function
def f12(x):
    return np.sqrt(np.absolute(np.float_power(np.sum(x*x,axis=0),2)-np.float_power(np.sum(x,axis=0),2))) + (0.5*np.sum(x*x,axis=0) + np.sum(x,axis=0))/D + 0.5

#Expanded Griewank's plus Rosenbrock's Function
def f13(x):
    res = 0
    for i in range(D-1):
        res+=auxf7(auxf4(np.array([x[i],x[i+1]])))
    res+=auxf7(auxf4(np.array([x[-1],x[0]])))
    return res

#Expanded Scaffer's F6 Function
def f14(x):
    res = 0
    for i in range(D-1):
        res+=auxg(x[i],x[i+1])
    res+=auxg(x[-1],x[0])
    return res
