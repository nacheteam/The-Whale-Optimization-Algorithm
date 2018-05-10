import numpy as np
import scipy

MIN_BOUND = -100
MAX_BOUND = 100
D=10

#Para la función 5
e = 2.718281

#Para la función 6
a=0.5
b=3
kmax=20

def setDimension(dim):
    global D
    D=dim

def getMinBound():
    global MIN_BOUND
    return MIN_BOUND

def getMaxBound():
    global MAX_BOUND
    return MAX_BOUND

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
#Estoy usando a directamente sin llamar a la constante por la definición de la función lambda.
def f6(x):
    x_mat = np.cos((np.repeat(0.5,(kmax+1)*D).reshape(kmax+1,D) + np.repeat(x,kmax+1).reshape(kmax+1,D))*2*scipy.pi*np.repeat(np.float_power(np.repeat(b,kmax+1),range(kmax+1)),D).reshape(kmax+1,D))
    ak_mat = np.repeat(np.fromfunction(lambda i,j: np.float_power(0.5,i*j+j),(1,kmax+1))[0],D).reshape(kmax+1,D)
    bk = np.cos(0.5*2*scipy.pi*np.fromfunction(lambda i,j: np.float_power(3,i*j+j),(1,kmax+1))[0])
    ak = np.fromfunction(lambda i,j: np.float_power(0.5,i*j+j),(1,kmax+1))[0]
    return np.sum(np.sum(x_mat*ak_mat,axis=1)) - D*np.sum(ak*bk)

#Griewank's Function
def f7(x):
    return np.sum(x*x)*(1.0/4000) - np.prod(np.divide(np.cos(x),np.sqrt(np.arange(1,D+1)))) + 1
