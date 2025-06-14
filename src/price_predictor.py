import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class PricePredictor:
    """
    Clase para predecir precios de productos utilizando modelos de regresión.

    Esta clase incluye funcionalidades para preprocesamiento (remoción de outliers),
    entrenamiento, evaluación, validación cruzada y predicción de precios con regresores
    como Random Forest y XGBoost.

    Atributos:
        df (pd.DataFrame): DataFrame de entrada con las características y la variable objetivo.
        id_column (pd.Series): Columna de IDs de cada publicación.
        X (pd.DataFrame): Variables predictoras.
        y (pd.Series): Variable objetivo original (precio).
        y_log (pd.Series): Logaritmo natural del precio, usado como target en los modelos.
        models (dict): Diccionario de modelos a entrenar.
        best_model (Regressor): Modelo seleccionado como el mejor tras la evaluación.
    """

    def __init__(self, df):
        """
        Inicializa el predictor con los datos y define los modelos base.

        Args:
            df (pd.DataFrame): DataFrame con las columnas 'id', 'price' y las variables predictoras.
        """
        self.df = df.copy()
        self.id_column = df['id']
        self.X = df.drop(columns=['id', 'price'])
        self.y = df['price']
        self.y_log = np.log1p(self.y)  # Transformación logarítmica para suavizar la distribución
        self.models = {
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'XGBoost': XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        }
        self.best_model = None

    def remove_outliers(self):
        """
        Elimina outliers del target transformado (log(precio)) usando el método del IQR.
        """
        Q1 = self.y_log.quantile(0.25)
        Q3 = self.y_log.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        mask = self.y_log.between(lower_bound, upper_bound)
        self.X = self.X[mask]
        self.y_log = self.y_log[mask]

    def train_test_split(self):
        """
        Divide los datos en conjuntos de entrenamiento y prueba (80/20).
        """
        self.X_train, self.X_test, self.y_train_log, self.y_test_log = train_test_split(
            self.X, self.y_log, test_size=0.2, random_state=42
        )

    def train_models(self):
        """
        Entrena todos los modelos definidos en `self.models` con los datos de entrenamiento.
        """
        self.trained_models = {}
        for name, model in self.models.items():
            model.fit(self.X_train, self.y_train_log)
            self.trained_models[name] = model

    def evaluate_model(self, model, name='Modelo'):
        """
        Evalúa un modelo sobre el conjunto de prueba y muestra métricas de rendimiento.

        Args:
            model: Modelo de regresión entrenado.
            name (str): Nombre del modelo para mostrar en consola.

        Returns:
            tuple: (MAE, RMSE, R²)
        """
        y_pred_log = model.predict(self.X_test)
        y_pred = np.expm1(y_pred_log)
        y_test = np.expm1(self.y_test_log)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        print(f"\nEvaluación de {name}:")
        print(f"MAE:  {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"R²:   {r2:.4f}")
        return mae, rmse, r2

    def cross_validate(self, model, name='Modelo'):
        """
        Realiza validación cruzada (5 folds) para un modelo usando RMSE como métrica.

        Args:
            model: Modelo de regresión a validar.
            name (str): Nombre del modelo.

        Returns:
            np.ndarray: Puntuaciones de validación cruzada (neg_root_mean_squared_error).
        """
        scores = cross_val_score(model, self.X, self.y_log, cv=5, scoring='neg_root_mean_squared_error')
        print(f"\nValidación cruzada para {name}:")
        print(f"RMSE medio: {-scores.mean():.2f} (+/- {scores.std():.2f})")
        return scores

    def compare_models(self):
        """
        Compara todos los modelos entrenados en base a su RMSE y selecciona el mejor.

        Returns:
            tuple: (Nombre del mejor modelo, métricas de rendimiento)
        """
        metrics = {}
        for name, model in self.trained_models.items():
            mae, rmse, r2 = self.evaluate_model(model, name)
            metrics[name] = {'mae': mae, 'rmse': rmse, 'r2': r2}

        best_model_name = min(metrics, key=lambda x: metrics[x]['rmse'])
        self.best_model = self.trained_models[best_model_name]
        print(f"\n✅ Modelo seleccionado: {best_model_name}")
        return best_model_name, metrics[best_model_name]

    def predict(self, X_new):
        """
        Predice precios para nuevas observaciones utilizando el mejor modelo entrenado.

        Args:
            X_new (pd.DataFrame): Nuevos datos con las mismas características que los datos de entrenamiento.

        Returns:
            np.ndarray: Predicciones en escala original de precios.
        """
        if self.best_model is None:
            raise ValueError("Primero debes entrenar y seleccionar el mejor modelo.")
        y_pred_log = self.best_model.predict(X_new)
        return np.expm1(y_pred_log)
