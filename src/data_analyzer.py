import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class DataAnalyzer:
    """
    Clase para realizar análisis exploratorio de datos sobre un dataset de productos.
    """

    def __init__(self, file_path):
        """
        Inicializa la clase con la ruta al archivo CSV.

        Parámetros:
        - file_path (str): Ruta al archivo CSV.
        """
        self.file_path = file_path
        self.df = None

    def read_data(self):
        """
        Lee el dataset desde un archivo CSV y filtra por vendedores en Argentina.
        """
        self.df = pd.read_csv(self.file_path)
        self.df = self.df[self.df['seller_country'] == 'Argentina']

    def summary_statistics(self):
        """
        Devuelve un resumen estadístico de los precios y cantidades vendidas.

        Retorna:
        - DataFrame con estadísticas descriptivas de 'price' y 'sold_quantity'.
        """

        print(self.df[['price', 'sold_quantity']].describe())
        return self.df[['price', 'sold_quantity']].describe()

    def handle_missing_values(self):
        """
        Identifica y muestra valores faltantes o inconsistentes. 
        También puede realizar una imputación simple o eliminar filas si es necesario.

        Retorna:
        - Conteo de valores nulos por columna.
        """
        missing = self.df.isnull().sum()
        print("Valores nulos por columna:\n", missing)
        
        return missing

    def detect_price_outliers(self, method="iqr", plot=True):
        """
        Detecta outliers en la columna 'price' usando el método IQR o Z-score.

        Parámetros:
        - method (str): Método para detectar outliers ('iqr' o 'zscore').
        - plot (bool): Si True, muestra un boxplot de precios.

        Retorna:
        - DataFrame con los productos considerados outliers.
        """
        if method == "iqr":
            Q1 = self.df['price'].quantile(0.25)
            Q3 = self.df['price'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = self.df[(self.df['price'] < lower_bound) | (self.df['price'] > upper_bound)]
        elif method == "zscore":
            mean = self.df['price'].mean()
            std = self.df['price'].std()
            z_scores = (self.df['price'] - mean) / std
            outliers = self.df[np.abs(z_scores) > 3]
        else:
            raise ValueError("Método no reconocido. Usa 'iqr' o 'zscore'.")

        num_outliers = outliers.shape[0]
        total = self.df.shape[0]
        porcentaje = (num_outliers / total) * 100 if total > 0 else 0

        print(f"\nNúmero de outliers detectados usando el método '{method}': {num_outliers}")
        print(f"Porcentaje de outliers respecto al total: {porcentaje:.2f}%")

        if num_outliers == 0:
            print("No se detectaron outliers con el método seleccionado.")
            

        if plot:
            sns.boxplot(x=self.df['price'])
            plt.title('Boxplot de precios')
            plt.show()

        return outliers