import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns

#%% Subida de variables y nombres

Variables = ["Patentes", "PIB Per Capita", "Inversión en I+D", "Calidad institucional", "Penetracion de internet"]
num_variables = len(Variables)

weights = np.array([0.87381757, 0.06854012, 0.03221508, 0.02044584, 0.00498138]) #Modelo obtenido por PCA
weights /= np.sum(weights)

np.random.seed(171239)  #Seed para reproducibilidad de los datos

data = pd.read_excel("Anexo 2 - Datos depurados.xlsx")    #Cambiar el anexo dependiendo de los datos que se deseen analizar, Anexo 2 para resultados, Anexo 3 para validación
logr = np.log(1+data.pct_change()[1:])

#%% analisis estadistico

m = logr.mean()
var = logr.var()
volatilidad = m -(0.5*var)
covar = logr.cov()
stdev = logr.std()


trials = 10000
years = 11          #se debe tener en cuenta que se debe ingresar el año n+1

Simulaciones = np.full(shape=(years, trials), fill_value=0.0)

Patentes_inic = data["Patentes"].iloc[-1]  #Valor inicial de patentes (año 2020), se puede cambiar el valor de iloc para iniciar en un año diferente
L = np.linalg.cholesky(covar)
u = norm.ppf(np.random.rand(num_variables, num_variables))
Lu = L.dot(u)    #Correlación de Cholesky

for i in range(0, trials):
    Z = norm.ppf(np.random.rand(years, num_variables))
    valores_anuales = np.inner(Lu, volatilidad.values + stdev.values*Z)
    Simulaciones[:,i] = np.cumprod(np.inner(weights, valores_anuales.T)+1)*Patentes_inic
    Simulaciones[0] = Patentes_inic
0

plt.figure(figsize=(15,8))
plt.plot(Simulaciones)
plt.ylabel("Numero de patentes")
plt.xlabel("Años")
plt.title("Simulación MC con " + str(trials) + " simulaciones " + str(Variables) + "\n" + str(np.round(weights*100,2)))
plt.show()
plt.figure(figsize=(20,16))
sns.displot(pd.DataFrame(Simulaciones).iloc[-1])
plt.ylabel("Frecuencia")
plt.xlabel("Numero de Patentes")
plt.show()

print("------------Media de la simulación------------")
print(Simulaciones.mean())
print("------------desviación estandar------------")
print(Simulaciones.std())
print("------------Maximo------------")
print(Simulaciones.max())
print("------------Minimo------------")
print(Simulaciones.min())
print("------------99%------------")
print(np.percentile(Simulaciones, 99))   #Se puede modificar el segundo valor a fin de modificar los percentiles
print("------------1%------------")
print(np.percentile(Simulaciones, 1))
print("------------95%------------")
print(np.percentile(Simulaciones, 95))
print("------------5%------------")
print(np.percentile(Simulaciones, 5))
print("------------Cuartil 1------------")
print(np.percentile(Simulaciones, 25))
print("------------Cuartil 2------------")
print(np.percentile(Simulaciones, 50))
print("------------Cuartil 3------------")
print(np.percentile(Simulaciones, 75))
print("------------MAtriz L------------")
print(L)
print("------------MAtriz Lu------------")
print(Lu)



d