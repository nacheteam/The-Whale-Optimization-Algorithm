import ballena
import sys
sys.path.insert(0, './funciones')
import funciones_objetivo

ballena.setSeed(123456789)
ballenas = ballena.getFuncionesBallena()

for i in range(len(ballenas)):

    print("Versión de la ballena " + str(i+1))

    fichero = open("./resultados/v{0}/resultados.txt".format(i+1),"w")

    funciones = funciones_objetivo.getFunciones()
    for dim in [10,30,50,100]:
        print("DIMENSION: " + str(dim))
        fichero.write("################################################\n")
        fichero.write("                    DIMENSION " + str(dim) + "\n")
        fichero.write("################################################\n")
        fichero.write("\n")
        contador = 1
        for f in funciones:
            solucion,fitness = ballenas[i](f,-100,100,dim)
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
