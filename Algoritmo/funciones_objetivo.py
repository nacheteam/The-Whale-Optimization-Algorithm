import numpy as np

MIN_BOUND = -100
MAX_BOUND = 100
D=10

def setDimension(dim):
    global D
    D=dim

def getMinBound():
    global MIN_BOUND
    return MIN_BOUND

def getMaxBound():
    global MAX_BOUND
    return MAX_BOUND

def f1(x):
    global D
    return np.sum(np.float_power(np.repeat(1000000,D),(range(1,D+1)-np.ones(D))/(np.repeat(D,D)-np.ones(D))) * (x*x))

def f2(x):
    global D
    return np.sum(np.append(1,np.repeat(1000000,D-1))*x*x)

def f3(x):
    global D
    return np.sum(np.append(1000000,np.repeat(1,D-1))*x*x)
