#%%
# carga de paquetes 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
# %%
df = pd.read_csv('logit_3.csv')
# %%
df.describe()
# %%
df.head()
# %%
# crear categorías
df['PasajeroFrec']   = df['PasajeroFrec'].astype('category')
df['IngresoAnual']   = df['IngresoAnual'].astype('category')
df['Hotel']          = df['Hotel'].astype('category')
print(df.dtypes)
# %%
df['PasajeroFrec']  = df['PasajeroFrec'].cat.codes
df['IngresoAnual']  = df['IngresoAnual'].cat.codes
df['Hotel']         = df['Hotel'].cat.codes

df.head()
# %%
X = df.drop('Target', axis = 1)
y = df['Target']
# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
# %%
model_log = LogisticRegression()
model_log.fit(X_train,y_train)
model_log.score(X_train, y_train)
model_log.score(X_test, y_test)
# %%
# SVM
model_svm = LinearSVC()
model_svm.fit(X_train,y_train)
model_svm.score(X_train, y_train)
model_svm.score(X_test, y_test)
# %%
