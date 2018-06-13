import ballena
import sys
sys.path.insert(0, './funciones')
import funciones_objetivo
import benchmark
import cma
import numpy as np

arr = np.random.uniform(0,1,10)
stdv = np.std(arr)
print(stdv)

es = cma.CMAEvolutionStrategy(arr,stdv,{'seed': 123456789,'verb_disp':0})
es.optimize(funciones_objetivo.getFuncion(1),verb_disp=0)
print(es.result)
