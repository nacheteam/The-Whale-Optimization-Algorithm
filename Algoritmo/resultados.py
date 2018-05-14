import funciones_objetivo
import ballena

fichero = open("./resultados/v1/resultados.txt","w")

funciones_objetivo.obtenerOptimosMatrices()
funciones = funciones_objetivo.getFunciones()
for dim in [10,30,50,100]:
    funciones_objetivo.setDimension(dim)
    funciones_objetivo.obtenerOptimosMatrices()
    print("DIMENSION: " + str(dim))
    fichero.write("################################################\n")
    fichero.write("                    DIMENSION " + str(dim) + "\n")
    fichero.write("################################################\n")
    fichero.write("\n")
    contador = 1
    for f in funciones:
        solucion,fitness = ballena.Ballena(f,funciones_objetivo.MIN_BOUND,funciones_objetivo.MAX_BOUND,dim)
        print("Función número " + str(contador))
        print("Solucion: " + str(solucion))
        print("Fitness: " + str(fitness))
        fichero.write("Función número " + str(contador) + "\n")
        fichero.write("Solucion: " + str(solucion) + "\n")
        fichero.write("Fitness: " + str(fitness) + "\n")
        fichero.write("\n")
        contador+=1
    fichero.write("\n\n\n")

fichero.close()
