
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.font_manager
from matplotlib import style
style.use('ggplot') or plt.style.use('ggplot')
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale
import warnings

#%% se importan y se describen los datos
warnings.filterwarnings('ignore')

datos = pd.read_excel("Anexo 2 - Datos depurados.xlsx")
datos.info()

print('Varianza de cada variable')
print(datos.var(axis=0))

#%% Entrenamiento modelo PCA con escalado de los datos

pca_pipe = make_pipeline(StandardScaler(), PCA())
pca_pipe.fit(datos)

# Se extrae el modelodel pipeline
modelo_pca = pca_pipe.named_steps['pca']

#%% Se combierte el array a dataframe para añadir nombres a los ejes.
pd.DataFrame( data=modelo_pca.components_, columns=datos.columns, index=['Patentes', 'PIB per capita', 'Inversion i+d', 'Calidad institucional', "Penetración de internet"])



fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 2))
componentes = modelo_pca.components_
plt.imshow(componentes.T, cmap='viridis', aspect='auto')
plt.yticks(range(len(datos.columns)), datos.columns)
plt.xticks(range(len(datos.columns)), np.arange(modelo_pca.n_components_) + 1)
plt.grid(False)
plt.colorbar()
plt.show()

#%% Porcentaje de varianza explicada por cada componente


print('Porcentaje de varianza explicada por cada componente')

print(modelo_pca.explained_variance_ratio_)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
ax.bar(x=np.arange(modelo_pca.n_components_) + 1, height=modelo_pca.explained_variance_ratio_)

for x, y in zip(np.arange(len(datos.columns)) + 1, modelo_pca.explained_variance_ratio_):
    label = round(y, 2)
    ax.annotate(label, (x, y), textcoords="offset points", xytext=(0,10), ha='center')


ax.set_xticks(np.arange(modelo_pca.n_components_) + 1)
ax.set_ylim(0, 1.1)
ax.set_title('Proporción de varianza explicada por cada componente')
ax.set_xlabel('Componente principal')
ax.set_ylabel('Proporción varianza explicada')
plt.show()

# Porcentaje de varianza explicada acumulada

prop_varianza_acum = modelo_pca.explained_variance_ratio_.cumsum()

print('Porcentaje de varianza explicada acumulada')

print(prop_varianza_acum)


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
ax.bar(x=np.arange(modelo_pca.n_components_) + 1, height=modelo_pca.explained_variance_ratio_)

for x, y in zip(np.arange(len(datos.columns)) + 1, modelo_pca.explained_variance_ratio_):
    label = round(y, 2)
    ax.annotate(label, (x, y), textcoords="offset points", xytext=(0, 10), ha='center')

ax.set_xticks(np.arange(modelo_pca.n_components_) + 1)
ax.set_ylim(0, 1.1)
ax.set_title('Porcentaje de varianza explicada por cada componente')
ax.set_xlabel('Componente principal')
ax.set_ylabel('Por. varianza explicada')

# Porcentaje de varianza explicada acumulada

prop_varianza_acum = modelo_pca.explained_variance_ratio_.cumsum()
print('Porcentaje de varianza explicada acumulada')
print(prop_varianza_acum)
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
ax.plot(np.arange(len(datos.columns)) + 1, prop_varianza_acum, marker='o')

for x, y in zip(np.arange(len(datos.columns)) + 1, prop_varianza_acum):
    label = round(y, 2)
    ax.annotate(label, (x, y), textcoords="offset points", xytext=(0, 10), ha='center')

ax.set_ylim(0, 1.1)
ax.set_xticks(np.arange(modelo_pca.n_components_) + 1)
ax.set_title('Proporción de varianza explicada acumulada')
ax.set_xlabel('Componente principal')
ax.set_ylabel('Por. varianza acumulada')
plt.show()

