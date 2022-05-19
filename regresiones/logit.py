
#%%
# carga de paquetes 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
# %%
df = pd.read_csv('logit_1.csv')
# %%
df.head()
#%%
df.shape
# %%
plt.figure()
plt.scatter(df['edad'],df['compra'])
#%%
X = df.drop("compra",axis=1)   #Feature Matrix
y = df["compra"]          #Target Variable
# %%
model_reg = LinearRegression()
model_reg.fit(X,y)
# %%
# Ajuste
r_sq = model_reg.score(X, y)
print(f"R cuadrado: {r_sq}")
# %%
y_pred = model_reg.predict(X)
#%%
y_pred2 = y_pred > .5
y_pred2 = y_pred2.astype(int)
diff = y_pred2 == y
diff = diff.astype(int)
print(sum(diff)/len(y))

# %%
plt.figure()
plt.scatter(X,y_pred2)
# %%
model_log = LogisticRegression()
model_log.fit(X,y)
# %%
y_pred_log = model_log.predict(X)
# %%
# %%
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(X,y_pred_log, s=10, c='b', marker="s", label='first')
ax1.scatter(X,y_pred2, s=10, c='r', marker="o", label='second')
plt.legend(loc='upper left');
plt.show()
# %%
prob_pred = model_log.predict_proba(X)
fig = plt.figure()
plt.scatter(X, prob_pred[:,1])

#%%
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(X,y_pred, s=10, c='b', marker="s", label='first')
ax1.scatter(X,prob_pred[:,1], s=10, c='r', marker="o", label='second')
plt.legend(loc='upper left');
plt.show()


# %%
model_log.score(X, y)
# %%
