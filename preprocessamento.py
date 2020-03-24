import numpy as np
import pandas as pd

# importando o dataset
dataset = pd.read_csv('datasets/dados_aula01.csv')

# Matrix de caracteristicas(variaveis independentes)
# Array/Matriz (variaveis dependentes)

# matrix de características (variáveis independentes)
X = dataset.iloc[:, :-1].values
print(X)
Y = dataset.iloc[:, 3]
print(Y)
