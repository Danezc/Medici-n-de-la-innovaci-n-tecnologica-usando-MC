import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns
from matplotlib import style

style.use('seaborn')
# Se toma el anexo "Datos finales.xlsx" y se deja unicamente la variable sobre la cual se realizará la simulación,
# Esta se debe ingresar en la linea 13 el nombre de la variable a con esto en cuenta, se cambia el nombre de las graficas y las variables a fin de dar sentido a el código
Variables = "PIB per capita"
df = pd.read_excel("Anexo 2 - Datos depurados.xlsx")
data = pd.DataFrame(df["PIB_per_capita"])

log_returns = np.log(1+data.pct_change())
np.random.seed(171239)
u = log_returns.mean()
var = log_returns.var()
volat = u - (0.5*var)
stdev = log_returns.std()
years = 2   # se debe poner el año n + 1 al valor que se desea conocer
trials = 50000 #Numero de simulaciones basadas en analisis de estabilidad

Z = norm.ppf(np.random.rand(years, trials))
Valoresanuales = np.exp(volat.values + stdev.values * Z)
Simulaciones = np.zeros_like(Valoresanuales)
Simulaciones[0] = data.iloc[-1]
for t in range(1, years):
    Simulaciones[t] = Simulaciones[t-1]*Valoresanuales[t]  #Si el valor esta en porcentaje se recomienda poner un "#" al comienzo de la línea y quitar los "#" de la línea 30 a 38

    #valor = Simulaciones[t-1]*Valoresanuales[t]
    #valor_2 = []
    #for i in valor:
        #if i > 1:
            #valor_2.append(1)
        #else:
            #valor_2.append(i)
    #Simulaciones[t] = valor_2

plt.figure(figsize=(17,6))
plt.title("Evolución penetración de inetrnet con Monte Carlo")
plt.plot(pd.DataFrame(Simulaciones))
plt.xlabel("Número de años")
plt.ylabel(Variables)
plt.show()

sns.displot(pd.DataFrame(Simulaciones).iloc[-1])

plt.xlabel(Variables + " a " + str(years - 1) + " años")
plt.ylabel("Frecuencia")
plt.show()

print(Simulaciones.mean())
print(Simulaciones.std())
print(Simulaciones.max())
print(Simulaciones.min())
print(np.percentile(Simulaciones, 99))   #Se puede modificar el segundo valor a fin de modificar los percentiles
print(np.percentile(Simulaciones, 1))
print(np.percentile(Simulaciones, 25))
print(np.percentile(Simulaciones, 50))
print(np.percentile(Simulaciones, 75))
