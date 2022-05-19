#%%
# carga de paquetes 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# %%
df = pd.read_csv('precio_casas.csv')
# %%
df.head()
# %%
one_hot = pd.get_dummies(df['Type'])
# %%
# Drop column B as it is now encoded
df = df.drop('Type',axis = 1)
# %%
df = df.join(one_hot)

# %%
df.head()
# %%
df = df.drop('u',axis = 1)

#%%
X = df.drop("Price",axis=1)   #Feature Matrix
y = df["Price"] /10000         #Target Variable
# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# %%
model = LinearRegression()
model.fit(X_train,y_train)
# %%
# Ajuste
r_sq = model.score(X_train, y_train)
print(f"R cuadrado: {r_sq}")
# %%
# Ajuste
r_sq = model.score(X_test, y_test)
print(f"R cuadrado: {r_sq}")
# %%
