import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Datos de la accion que queramos
data = yf.download('AAPL', start='2012-01-01', end='2022-01-01')
data['Date'] = pd.to_datetime(data['Date'])
data.to_csv('./datos/Test.csv', index=False, date_format='%Y-%m-%d')

# Variables predictoras
data['SMA_10'] = data['Close'].rolling(window=10).mean()
data['SMA_30'] = data['Close'].rolling(window=30).mean()
data['SMA_60'] = data['Close'].rolling(window=60).mean()
data['SMA_100'] = data['Close'].rolling(window=100).mean()
data = data.dropna()
x = data[['SMA_10', 'SMA_30', 'SMA_60', 'SMA_100']].values

# Crear la variable objetivo
y = data['Close'].values

# Datos de entrenamiento y prueba
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=0) 

# Modelo de regresion lineal
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Predecir precios
prediction = regressor.predict(X_test)

# Evaluar precision del modelo
mse = mean_squared_error(Y_test, prediction)
print('Error cuadrático medio: ', mse)

datex = data.tail(90)
fig, ax = plt.subplots(figsize=(12,6))
ax.plot(datex.index, datex['Close'], label='precio real')
ax.plot(datex.index, prediction[-90:], label='precio predicho')

ax.set_title('evolucion del precio')
ax.set_xlabel('fecha')
ax.set_ylabel('precio')

ax.legend()
plt.show()


# Graficar
# plt.scatter(Y_test, prediction, alpha=0.5)
# plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'k--', lw=2)
# plt.xlabel('Valor Real')
# plt.ylabel('Valor Predicho')
# plt.show()