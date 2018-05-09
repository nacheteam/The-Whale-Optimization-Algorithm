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
