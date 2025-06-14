# Predicción de Ventas y Precios en Publicaciones de Mercado Libre

Este repositorio contiene un proyecto de análisis de datos y machine learning desarrollado como parte de una prueba técnica para Mercado Libre. El objetivo es predecir el **precio ideal** y la **probabilidad de venta** de publicaciones en el marketplace, a partir de sus características descriptivas.

## 📁 Estructura del Proyecto
├── data/
│ └── new_items_dataset.csv # (No incluido por tamaño)
├── src/
│ ├── main.py # Script principal de ejecución
│ ├── data_analyzer.py # Exploración y limpieza de datos
│ ├── feature_engineering.py # Ingeniería de características
│ ├── price_predictor.py # Modelado de regresión de precios
│ └── sales_predictor.py # Modelado de clasificación de ventas
├── requirements.txt
└── README.md

## 🚀 ¿Qué hace este proyecto?

1. **Explora y limpia el dataset** (`DataAnalyzer`)
2. **Transforma las variables** para análisis predictivo (`FeatureEngineer`)
3. **Predice el precio óptimo** usando modelos de regresión (`PricePredictor`)
4. **Clasifica la probabilidad de venta** con modelos de clasificación y técnicas para datos desbalanceados (`SalesPredictor`)
5. **Valida y compara modelos** aplicando técnicas como validación cruzada y ajuste de umbral de decisión.

## ⚙️ Instalación

1. Clona este repositorio:

git clone https://github.com/tu_usuario/nombre_repositorio.git
cd nombre_repositorio

2. Crea un entorno virtual y activa (opcional pero recomendado)

python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

3. Instala las dependencias:

pip install -r requirements.txt

📂 Dataset
El archivo data/new_items_dataset.csv no está incluido en el repositorio debido a restricciones de tamaño en GitHub (>100MB). Para ejecutar el proyecto, debes colocar manualmente este archivo en la carpeta data/

▶️ Ejecución
Desde la raíz del proyecto:
python src/main.py
El script ejecutará automáticamente todo el pipeline: análisis exploratorio, procesamiento, entrenamiento de modelos y evaluación.

📈 Métricas
El proyecto calcula e imprime métricas relevantes para ambos tipos de modelos: MAE y RMSE en regresión, y precisión, recall, F1-score en clasificación.

📬 Contacto
Lina Marcela Aguilar
lina.m.aguilar@gmail.com