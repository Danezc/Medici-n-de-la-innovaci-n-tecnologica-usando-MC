import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_excel("Anexo 2 - Datos depurados.xlsx")
df.describe().to_csv("Descriptivas.csv")

print(np.std(df))

plt.plot(df["Año"], df["Penetracion_de_internet"])
plt.ylabel("Penetración de internet")
plt.xlabel("Año")
plt.xticks(range(2000, 2021, 2))
plt.title("Penetración de internet")
plt.show()
