import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Predictor IA de Acciones", layout="wide", page_icon="🔮")

# Título Estilizado
st.title("🔮 Oráculo Financiero: IA Predictiva")
st.markdown("""
Esta aplicación utiliza **Machine Learning** (Regresión Logística) para analizar el comportamiento
pasado de una acción y predecir si el precio **SUBIRÁ** o **BAJARÁ** mañana.
""")

# --- BARRA LATERAL (INPUTS) ---
st.sidebar.header("⚙️ Configuración")
ticker = st.sidebar.text_input("Símbolo (Ticker):", value="AAPL")

# --- FUNCIÓN DE CARGA DE DATOS ---
def descargar_datos(ticker):
    try:
        df = yf.download(ticker, period="2y", progress=False) # 2 años de historia para entrenar
        
        # Corrección para yfinance reciente (MultiIndex)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            df = df.reset_index()
            df = df.set_index('Date')
            
        return df
    except Exception as e:
        return None

# --- LÓGICA DEL CEREBRO (IA) ---
def entrenar_modelo(df):
    data = df.copy()
    
    # 1. Crear Variables (Feature Engineering)
    data['Retorno'] = data['Close'].pct_change()
    data['Lag_1'] = data['Retorno'].shift(1) # Qué pasó ayer
    data['Lag_2'] = data['Retorno'].shift(2) # Qué pasó antier
    data['Volatilidad'] = data['Close'].rolling(5).std() # Qué tan loco está el mercado
    data['Momentum'] = data['Close'] - data['Close'].rolling(10).mean() # Tendencia
    
    data = data.dropna()
    
    # 2. Definir Objetivo (1: Sube, 0: Baja)
    data['Target'] = np.where(data['Retorno'].shift(-1) > 0, 1, 0)
    
    # 3. Entrenar
    features = ['Lag_1', 'Lag_2', 'Volatilidad', 'Momentum']
    X = data[features]
    y = data['Target']
    
    # Usamos todos los datos menos el último día para entrenar
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Medir precisión
    precision = model.score(X_test, y_test)
    
    return model, features, precision, data

# --- EJECUCIÓN PRINCIPAL ---
df = descargar_datos(ticker)

if df is not None and not df.empty:
    # 1. MOSTRAR DATOS
    col1, col2 = st.columns(2)
    precio_actual = df['Close'].iloc[-1]
    col1.metric("Precio Actual", f"${precio_actual:.2f}")
    
    # Gráfico simple
    st.subheader(f"Gráfico de Precios: {ticker}")
    st.line_chart(df['Close'])
    
    # 2. ACTIVAR LA IA
    st.markdown("---")
    st.subheader("🧠 Análisis del Algoritmo")
    
    with st.spinner('Entrenando modelo en tiempo real...'):
        modelo, features, precision, data_procesada = entrenar_modelo(df)
    
    # Mostrar Precisión del Robot
    st.info(f"📊 Precisión histórica del modelo para {ticker}: **{precision:.1%}**")
    
    # 3. PREDICCIÓN PARA MAÑANA
    # Tomamos los datos de HOY para predecir MAÑANA
    ultimo_dia = data_procesada.iloc[[-1]][features]
    prediccion = modelo.predict(ultimo_dia)
    probabilidad = modelo.predict_proba(ultimo_dia)
    
    # Probabilidad de Subir (Clase 1)
    prob_subir = probabilidad[0][1]
    
    st.markdown("### 🔮 Predicción para Mañana:")
    
    col_pred, col_conf = st.columns(2)
    
    if prob_subir > 0.5:
        col_pred.success("🚀 EL MODELO DICE: **SUBIRÁ**")
        color = "green"
    else:
        col_pred.error("🔻 EL MODELO DICE: **BAJARÁ**")
        color = "red"
        
    col_conf.metric("Nivel de Confianza (Probabilidad)", f"{prob_subir:.1%}")

    # Explicación
    st.caption("Nota: La 'Confianza' indica qué tan seguro está el modelo matemático. Más del 50% indica tendencia alcista.")

else:
    st.warning("No se encontraron datos. Revisa el símbolo de la acción.")


    