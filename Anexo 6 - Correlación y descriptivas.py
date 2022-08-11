import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

df = pd.read_excel("Anexo 2 - Datos depurados.xlsx")  #Se debe cambiar el nombre de la base de datos según la que se quieran conocer las estadisticas descriptivas

sns.pairplot(df)
plt.show()
sns.heatmap(df.corr(), vmin=-1, vmax=+1, annot=True, cmap='coolwarm')
plt.show()

print(df.describe())



