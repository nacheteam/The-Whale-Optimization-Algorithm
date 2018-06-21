#!/bin/usr/gnuplot

################################################################################
#                          GRAFICAS PROGRESION                                 #
################################################################################

set title "Progresion D10"
set auto x
set yrange [0:100000]
plot "./progresion_mejores_10.txt" with lines title "Progresion F1 D10"

set term png
set output "./Imagenes/Progresion/progresion_mejores_10.png"
replot
set term x11

set title "Progresion D30"
set auto x
set yrange [0:700000]
plot "./progresion_mejores_30.txt" with lines title "Progresion F1 D30"

set term png
set output "./Imagenes/Progresion/progresion_mejores_30.png"
replot
set term x11

################################################################################
#                               GRAFICAS ERRORES                               #
################################################################################

set title "Errores V1 D10"
set auto x
set yrange [0:20000]
set style data histogram
set style fill solid border -1
set boxwidth 0.9
set xtic rotate by -45 scale 0
plot "./v1/errores_d10.txt" using 2:xticlabels(1) title "Errores-D10 V1"

set term png
set output "./Imagenes/Errores/errores_v1_d10.png"
replot
set term x11

set title "Errores V1 D30"
set auto x
set yrange [0:200000]
set style data histogram
set style fill solid border -1
set boxwidth 0.9
set xtic rotate by -45 scale 0
plot "./v1/errores_d30.txt" using 2:xticlabels(1) title "Errores-D30 V1"

set term png
set output "./Imagenes/Errores/errores_v1_d30.png"
replot
set term x11

set title "Errores V2 D10"
set auto x
set yrange [0:40000]
set style data histogram
set style fill solid border -1
set boxwidth 0.9
set xtic rotate by -45 scale 0
plot "./v2/errores_d10.txt" using 2:xticlabels(1) title "Errores-D10 V2"

set term png
set output "./Imagenes/Errores/errores_v2_d10.png"
replot
set term x11

set title "Errores V2 D30"
set auto x
set yrange [0:300000]
set style data histogram
set style fill solid border -1
set boxwidth 0.9
set xtic rotate by -45 scale 0
plot "./v2/errores_d30.txt" using 2:xticlabels(1) title "Errores-D30 V2"

set term png
set output "./Imagenes/Errores/errores_v2_d30.png"
replot
set term x11

set title "Errores V3 D10"
set auto x
set yrange [0:13000]
set style data histogram
set style fill solid border -1
set boxwidth 0.9
set xtic rotate by -45 scale 0
plot "./v3/errores_d10.txt" using 2:xticlabels(1) title "Errores-D10 V3"

set term png
set output "./Imagenes/Errores/errores_v3_d10.png"
replot
set term x11

set title "Errores V3 D30"
set auto x
set yrange [0:100000]
set style data histogram
set style fill solid border -1
set boxwidth 0.9
set xtic rotate by -45 scale 0
plot "./v3/errores_d30.txt" using 2:xticlabels(1) title "Errores-D30 V3"

set term png
set output "./Imagenes/Errores/errores_v3_d30.png"
replot
set term x11

set title "Errores V4"
set auto x
set yrange [0:60000]
set style data histogram
set style fill solid border -1
set boxwidth 0.9
set xtic rotate by -45 scale 0
plot "./v4/errores_d10.txt" using 2:xticlabels(1) title "Errores-D10 V4", "./v4/errores_d30.txt" using 2:xticlabels(1) title "Errores-D30 V4"

set term png
set output "./Imagenes/Errores/errores_v4.png"
replot
set term x11

set title "Errores V5"
set auto x
set yrange [0:3500]
set style data histogram
set style fill solid border -1
set boxwidth 0.9
set xtic rotate by -45 scale 0
plot "./v5/errores_d10.txt" using 2:xticlabels(1) title "Errores-D10 V5", "./v5/errores_d30.txt" using 2:xticlabels(1) title "Errores-D30 V5"

set term png
set output "./Imagenes/Errores/errores_v5.png"
replot
set term x11

set title "Errores V6"
set auto x
set yrange [0:3000]
set style data histogram
set style fill solid border -1
set boxwidth 0.9
set xtic rotate by -45 scale 0
plot "./v6/errores_d10.txt" using 2:xticlabels(1) title "Errores-D10 V6", "./v6/errores_d30.txt" using 2:xticlabels(1) title "Errores-D30 V6"

set term png
set output "./Imagenes/Errores/errores_v6.png"
replot
set term x11

################################################################################
#                              GRAFICAS RESULTADOS                             #
################################################################################

set title "Resultados V1 D10"
set auto x
set yrange [0:20000]
set style data histogram
set style fill solid border -1
set boxwidth 0.9
set xtic rotate by -45 scale 0
plot "./v1/resultados_d10.txt" using 2:xticlabels(1) title "Resultados-D10 v1"

set term png
set output "./Imagenes/Resultados/resultados_v1_d10.png"
replot
set term x11

set title "Resultados V1 D30"
set auto x
set yrange [0:220000]
set style data histogram
set style fill solid border -1
set boxwidth 0.9
set xtic rotate by -45 scale 0
plot "./v1/resultados_d30.txt" using 2:xticlabels(1) title "Resultados-D30 v1"

set term png
set output "./Imagenes/Resultados/resultados_v1_d30.png"
replot
set term x11

set title "Resultados V2 D10"
set auto x
set yrange [0:40000]
set style data histogram
set style fill solid border -1
set boxwidth 0.9
set xtic rotate by -45 scale 0
plot "./v2/resultados_d10.txt" using 2:xticlabels(1) title "Resultados-D10 v2"

set term png
set output "./Imagenes/Resultados/resultados_v2_d10.png"
replot
set term x11

set title "Resultados V2 D30"
set auto x
set yrange [0:300000]
set style data histogram
set style fill solid border -1
set boxwidth 0.9
set xtic rotate by -45 scale 0
plot "./v2/resultados_d30.txt" using 2:xticlabels(1) title "Resultados-D30 v2"

set term png
set output "./Imagenes/Resultados/resultados_v2_d30.png"
replot
set term x11

set title "Resultados V3 D10"
set auto x
set yrange [0:15000]
set style data histogram
set style fill solid border -1
set boxwidth 0.9
set xtic rotate by -45 scale 0
plot "./v3/resultados_d10.txt" using 2:xticlabels(1) title "Resultados-D10 v3"

set term png
set output "./Imagenes/Resultados/resultados_v3_d10.png"
replot
set term x11

set title "Resultados V3 D30"
set auto x
set yrange [0:100000]
set style data histogram
set style fill solid border -1
set boxwidth 0.9
set xtic rotate by -45 scale 0
plot "./v3/resultados_d30.txt" using 2:xticlabels(1) title "Resultados-D30 v3"

set term png
set output "./Imagenes/Resultados/resultados_v3_d30.png"
replot
set term x11

set title "Resultados V4"
set auto x
set yrange [0:60000]
set style data histogram
set style fill solid border -1
set boxwidth 0.9
set xtic rotate by -45 scale 0
plot "./v4/resultados_d10.txt" using 2:xticlabels(1) title "Resultados-D10 v4", "./v4/resultados_d30.txt" using 2:xticlabels(1) title "Resultados-D30 v4"

set term png
set output "./Imagenes/Resultados/resultados_v4.png"
replot
set term x11

set title "Resultados V5"
set auto x
set yrange [0:4000]
set style data histogram
set style fill solid border -1
set boxwidth 0.9
set xtic rotate by -45 scale 0
plot "./v5/resultados_d10.txt" using 2:xticlabels(1) title "Resultados-D10 v5", "./v5/resultados_d30.txt" using 2:xticlabels(1) title "Resultados-D30 v5"

set term png
set output "./Imagenes/Resultados/resultados_v5.png"
replot
set term x11

set title "Resultados V6"
set auto x
set yrange [0:3500]
set style data histogram
set style fill solid border -1
set boxwidth 0.9
set xtic rotate by -45 scale 0
plot "./v6/resultados_d10.txt" using 2:xticlabels(1) title "Resultados-D10 v6", "./v6/resultados_d30.txt" using 2:xticlabels(1) title "Resultados-D30 v6"

set term png
set output "./Imagenes/Resultados/resultados_v6.png"
replot
set term x11
