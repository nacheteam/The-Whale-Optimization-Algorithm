import numpy as np
import scipy

MIN_BOUND = -100
MAX_BOUND = 100
D=10
e = 2.718281

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
