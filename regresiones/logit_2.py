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
from sklearn.datasets import make_blobs
import mglearn
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
import numpy as np
#%%
X, y = make_blobs(random_state=42)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.xlabel("Característica 0")
plt.ylabel("Característica 1")
plt.legend(["Clase 0", "Clase 1", "Clase 2"])
# %%
linear_svm = LinearSVC().fit(X, y)
print("Coefficient shape: ", linear_svm.coef_.shape)
print("Intercept shape: ", linear_svm.intercept_.shape)
# %%
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
line = np.linspace(-15, 15)
for coef, intercept, color in zip(linear_svm.coef_, linear_svm.intercept_,['b', 'r', 'g']):
    plt.plot(line, -(line * coef[0] + intercept) / coef[1], c=color)
    plt.ylim(-10, 15)
    plt.xlim(-10, 8)
    plt.xlabel("Característica 0")
    plt.ylabel("Característica 1")
    plt.legend(['Clase 0', 'Clase 1', 'Clase 2', 'Línea clase 0', 'Línea clase 1','Línea clase 2'], loc=(1.01, 0.3))
# %%
mglearn.plots.plot_2d_classification(linear_svm, X, fill=True, alpha=.7)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
line = np.linspace(-15, 15)
for coef, intercept, color in zip(linear_svm.coef_, linear_svm.intercept_,['b', 'r', 'g']):
    plt.plot(line, -(line * coef[0] + intercept) / coef[1], c=color)
    plt.legend(['Clase 0', 'Clase 1', 'Clase 2', 'Línea clase 0', 'Línea clase 1','Línea clase 2'], loc=(1.01, 0.3))
    plt.xlabel("Característica 0")
    plt.ylabel("Característica 1")
# %%
