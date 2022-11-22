import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def modelo(x, y, color, degree=1):
    # Ajuste del polinomio de grado 'degree' a los datos de entrenamiento x,y
    coeffs = np.polyfit(x, y, deg=degree)
    # Determinar y escribir la forma del polinomio
    p = np.poly1d(np.polyfit(x, y, deg=degree), variable='X')
    print("Polinomio de grado ", degree, " : ")
    print(p)
    print("")

    y_pred = np.polyval(np.poly1d(coeffs), x)
    print("Error cuadrático medio (ECM): ", 1/20*(sum((y-y_pred)**2)))
    print("")

    # Dibujar la gráfica del polinomio
    # Calcular la y de la gráfica 'y_plot'
    y_plot = np.polyval(np.poly1d(coeffs), x)

    # Dibujar la gráfica
    plt.plot(x, y_plot, color=color,
             linewidth=2, label="grado% d" % degree)


def obtener_leyenda(label="Leyenda", color="red") -> mpatches.Patch:
    return mpatches.Patch(facecolor=color, label=label,
                          linewidth=0.5, edgecolor='black')


data = pd.read_excel("IBEX35_Sept2018.xls")
df = pd.DataFrame(data, columns=["Dia", "Apertura", "Cierre"])
dias = df["Dia"].to_numpy()
apertura = df["Apertura"].to_numpy()
cierre = df["Cierre"].to_numpy()

plt.scatter(dias, apertura)
plt.scatter(dias, cierre)
plt.xticks(dias)
plt.xlabel("Septiembre")
plt.ylabel("Valor")

gradoApertura = 9
gradoCierre = 9

plt.legend(handles=[
    obtener_leyenda("Apertura", "blue"),
    obtener_leyenda("Cierre", "orange"),
    obtener_leyenda("Polinomio Apertura Grado: %(grado)x" % {"grado": gradoApertura}, "green"),
    obtener_leyenda("Polinomio Apertura Grado: %(grado)x" % {"grado": gradoCierre}, "purple")], title="Leyenda",
    loc=4, fontsize='small', fancybox=True)

modelo(dias, apertura, "green", gradoApertura)
modelo(dias, cierre, "purple", gradoCierre)

plt.show()
