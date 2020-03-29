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

# Qualidade da Informação
# Tratando dados "perdidos"

from sklearn.impute import SimpleImputer

dados_perdidos = SimpleImputer(missing_values=np.nan, strategy="mean", verbose=0)
# fit => usado para extrair as informações dos dados nos quais o objeto é aplicado
#   (quais os campos void e calcular média)
dados_perdidos = dados_perdidos.fit(X[:, 1:3])
X[:, 1:3] = dados_perdidos.transform(X[:, 1:3])

# Lidando com variaveis categorias
# Codificar os dados categoricos> substituir texto por números
#   tde modo a incluir essas variáveis a equação

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder

# Ordinal Encoding
code_X = ColumnTransformer([('codificar_X', OrdinalEncoder(), [0])], remainder='passthrough')
a = np.array(code_X.fit_transform(X))

# Dummy Encoding
from sklearn.preprocessing import OneHotEncoder

code_X = ColumnTransformer([('codificar_X', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(code_X.fit_transform(X))

# Como é uma variável dependente, o modelo vai saber que é categoria
# e não possui precedentes entre sim e não

from sklearn.preprocessing import LabelEncoder

code_y = LabelEncoder()
Y = code_y.fit_transform(Y)
# print(X)
# print(Y)

# Featre Scaling -> Escalar as caracteristicas
# Normalização e Padronização
# Quando devemos usar normalização  ou padronização?
# Geralemnte, voce deve normalizar quando os dados são
#   normalmente distribuido
# Na duvida usar a padronização


from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
# fit and transform
X = sc_X.fit_transform(X)

# Treinar o modelo para aprender e não decorar

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, train_size=0.8, random_state=0)
print(X_train)
print('----\n')
print(X_test)
