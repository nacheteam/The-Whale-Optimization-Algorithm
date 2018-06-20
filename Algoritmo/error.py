'''
Este script lee los ficheros de resultados y obtiene un fichero para cada uno
con el error cometido en cada función.
'''

for i in range(6):
    f = open("./resultados/v{}/resultados.txt".format(i+1))
    g = open("./resultados/v{}/errores.txt".format(i+1),"w")

    num_f = 1
    for line in f:
        if "Fitness" in line:
            fit = float(line.split("Fitness:")[1])
            g.write(str(num_f) + "\t" + str(fit-100*num_f) + "\n")
            num_f+=1
    g.close()
    f.close()
