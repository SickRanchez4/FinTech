import io
import base64
import csv
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from flask import Flask, render_template, request, Response
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# App init
app = Flask(__name__)

# Index
@app.route('/')
def index():
    return render_template('index.html')

# Prediccion
@app.route('/predecir')
def predecir():
    data = pd.read_csv('exporting_data.csv')

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
    #print('Error cuadrático medio: ', mse)

    # Graficar
    plt.scatter(Y_test, prediction, alpha=0.5)
    plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'k--', lw=2)
    plt.xlabel('Valor Real')
    plt.ylabel('Valor Predicho')

    # save the chart as an image in memory
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)

    # encode the image as a base64 string
    chart = base64.b64encode(img.read()).decode('utf-8')

    return render_template('predecir.html', file_contents=chart)

if __name__ == '__main__':
    app.run(debug=True)