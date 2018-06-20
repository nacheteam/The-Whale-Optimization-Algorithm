'''
Este script lee los ficheros de resultados y obtiene un fichero para cada uno
con el error cometido en cada función.
'''

for i in range(1):
    f = open("./v{}/resultados.txt".format(i+1))
    g = open("./v{}/errores.txt".format(i+1))

    for line in f:
        num_f = 1
        if "Fitness" in line:
            fit = float(line.split("Fitness:")[1])
            g.write(str(num_f) + "\t" + str(fit-100*num_f) + "\n")
    g.close()
    f.close()
