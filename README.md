# 🤖 Apple Stock Predictor & Algorithmic Trading Bot

## 📋 Project Overview
Este proyecto desarrolla un algoritmo de Inteligencia Artificial capaz de predecir la dirección del precio de las acciones de Apple (AAPL) y ejecutar una estrategia de trading simulada con gestión de riesgo.

A diferencia de los modelos tradicionales, este bot no solo busca maximizar ganancias, sino que **detecta la volatilidad del mercado** para proteger el capital en momentos de crisis.

## 🧠 Modelos y Estrategia
* **Regresión Logística:** Clasificación binaria para predecir movimientos diarios (Sube/Baja).
* **Feature Engineering:** Uso de indicadores técnicos (SMA 10/50, Volatilidad) y correlación con el S&P 500.
* **Gestión de Riesgo:** El algoritmo utiliza un *Umbral de Confianza Dinámico*. Solo opera cuando la probabilidad de éxito supera el promedio histórico, pasando a efectivo (Cash) durante alta incertidumbre.

## 📊 Resultados (Backtesting 2024-2025)
En las simulaciones de estrés, el algoritmo logró **evitar una caída del mercado del 20%** (Marzo 2025) al detectar el cambio de tendencia y salir de la posición automáticamente, demostrando capacidades de preservación de capital superiores a la estrategia "Buy & Hold".

## 🛠️ Stack Tecnológico
* Python (Pandas, NumPy)
* Scikit-Learn (Machine Learning)
* YFinance (Datos de mercado en tiempo real)
