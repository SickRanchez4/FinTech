import io
import base64
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from flask import Flask, render_template, request, Response
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from sklearn.metrics import mean_squared_error

# App init
app = Flask(__name__)

# Index
@app.route('/')
def index():
    return render_template('index.html')

# Prediccion
@app.route('/predecir', methods=['POST'])
def predecir():
    valor = request.form['Valor']
    periodo = request.form['Periodo']
    M1 = request.form['M1']
    M2 = request.form['M2']
    M3 = request.form['M3']
    M4 = request.form['M4']
    prediction = predict(valor, periodo, M1, M2, M3, M4)

    return render_template('predecir.html', file_contents=prediction[0], mse=prediction[1])

# Chart
@app.route('/chart')
def chart():
    return render_template('chart.html')

def predict(valor, periodo, media1, media2, media3, media4):
    stockName = getDataExtension(valor)
    data = yf.download(stockName, start='2012-01-01', end='2022-01-01')
    periodo = int(periodo)

    # Variables predictoras
    data['SMA_10'] = data['Close'].rolling(window=int(media1)).mean()
    data['SMA_30'] = data['Close'].rolling(window=int(media2)).mean()
    data['SMA_60'] = data['Close'].rolling(window=int(media3)).mean()
    data['SMA_100'] = data['Close'].rolling(window=int(media4)).mean()
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
    _mse = mean_squared_error(Y_test, prediction)
    _mse = round(_mse, 2)
    print('Error cuadrático medio: ', _mse)

    # Obtener datos del ultimo tiempo
    last_data = data.tail(periodo)

    # Figura y eje
    fig, ax = plt.subplots(figsize=(12,6))

    # Graficar para el ultimo tiempo
    ax.plot(last_data.index, last_data['Close'], label='Precio Real')
    ax.plot(last_data.index, prediction[-periodo:], label='Precio Predicho')
    ax.set_title('Evolución del precio de la acción en el último tiempo')
    ax.set_xlabel('Fecha')
    ax.set_ylabel('Precio USD')
    ax.legend()

    # guardar el chart como img
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)

    # encode imagen a base64
    chart = base64.b64encode(img.read()).decode('utf-8')

    return chart, _mse

def getDataExtension(valor):
    if valor == 'GOOGLE':
        return 'GOOG'
    elif valor =='MCDONALD':
        return 'MCD'
    elif valor =='META':
        return 'META'
    else : None
    
if __name__ == '__main__':
    app.run(debug=True)