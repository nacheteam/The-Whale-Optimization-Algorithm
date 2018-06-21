for i in range(6):
    f = open("./resultados/v{}/resultados.txt".format(i+1))
    g = open("./resultados/v{}/resultados_formateados.txt".format(i+1),"w")

    num_f = 1
    for line in f:
        if "Fitness" in line:
            fit = float(line.split("Fitness:")[1])
            if num_f<=20:
                g.write("f{}-D10".format(num_f) + "\t" + str(fit-100*num_f) + "\n")
            else:
                g.write("f{}-D30".format(num_f-20) + "\t" + str(fit-100*num_f) + "\n")
            num_f+=1
    g.close()
    f.close()
