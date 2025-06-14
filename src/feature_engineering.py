import pandas as pd
import numpy as np
import ast
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

class FeatureEngineer:
    """
    Clase para realizar ingeniería de características sobre un DataFrame del marketplace de Mercado Libre.
    Incluye eliminación de columnas, creación de nuevas variables, codificación de variables categóricas y transformación de datos.

    Parámetros:
    - df (pd.DataFrame): DataFrame original procesado por la clase DataAnalyzer.
    """
    def __init__(self, df):
        self.df = df.copy()
        self.le_category = LabelEncoder()
        self.le_seller = LabelEncoder()
        self.mlb_tags = MultiLabelBinarizer()

    def drop_unnecessary_columns(self):
        """
        Elimina columnas irrelevantes para el análisis o modelado.
        """
        cols = ['sub_status', 'attributes', 'variations', 'seller_country', 'seller_city', 'warranty']
        self.df.drop(columns=cols, inplace=True, errors='ignore')

    def add_price_diff_and_pictures_count(self):
        """
        Calcula la diferencia entre el precio base y el precio actual.
        También calcula la cantidad de imágenes por producto y elimina las columnas originales.

        Además, imprime el porcentaje de productos cuyo precio no ha cambiado respecto al base.
        """
        self.df['price_diff'] = self.df['price'] - self.df['base_price']
        self.df['picture_count'] = self.df['pictures'].apply(
            lambda x: len(ast.literal_eval(x)) if isinstance(x, str) and x.startswith('[') else 0
        )
        porcentaje_ceros = (self.df['price_diff'] == 0).mean() * 100
        print(f"Porcentaje de valores 0 en price_diff: {porcentaje_ceros:.2f}%")
        print("Por lo tanto no se tendrá en cuenta el base_price como feature ya que la mayoría de las veces es la misma variable objetivo")
        self.df.drop(columns=['base_price', 'pictures', 'price_diff'], inplace=True, errors='ignore')

    def encode_booleans(self):
        """
        Convierte columnas booleanas representadas como strings en valores booleanos reales.
        """
        self.df['shipping_admits_pickup'] = self.df['shipping_admits_pickup'].apply(lambda x: str(x).lower() == 'true')
        self.df['shipping_is_free'] = self.df['shipping_is_free'].apply(lambda x: str(x).lower() == 'true')
        self.df['is_new'] = self.df['is_new'].astype(bool)

    def add_title_features(self):
        """
        Agrega características relacionadas con el título del producto:
        - Longitud en caracteres.
        - Cantidad de palabras.
        Luego elimina la columna original 'title' porque no se usará NLP.
        """
        self.df['title_len'] = self.df['title'].str.len()
        self.df['title_word_count'] = self.df['title'].str.split().apply(len)
        self.df.drop(columns=['title'], inplace=True)

    def add_date_features(self):
        """
        Extrae información temporal de la fecha de creación del producto y calcula cuántos días han pasado desde su publicación.
        Luego elimina la columna original de fecha.
        """
        self.df['date_created'] = pd.to_datetime(self.df['date_created'], errors='coerce')
        self.df['date_created'] = self.df['date_created'].dt.tz_localize(None)
        self.df['year_created'] = self.df['date_created'].dt.year
        self.df['month_created'] = self.df['date_created'].dt.month
        self.df['day_created'] = self.df['date_created'].dt.day
        now_naive = pd.Timestamp.now(tz=None)
        self.df['days_since_creation'] = (now_naive - self.df['date_created']).dt.days
        self.df.drop(columns=['date_created'], inplace=True)

    def encode_categories(self):
        """
        Codifica las variables categóricas 'category_id' y 'seller_id' usando LabelEncoder.
        """
        self.df['category_id'] = self.le_category.fit_transform(self.df['category_id'])
        self.df['seller_id'] = self.df['seller_id'].fillna('missing').astype(str)
        self.df['seller_id_encoded'] = self.le_seller.fit_transform(self.df['seller_id'])
        self.df = self.df.drop(columns=['seller_id'], errors='ignore')

    def encode_dummies(self):
        """
        Aplica one-hot encoding a varias columnas categóricas para convertirlas en variables binarias.
        """
        categorical_cols = ['seller_province', 'seller_loyalty', 'buying_mode', 'shipping_mode', 'status']
        self.df = pd.get_dummies(self.df, columns=categorical_cols, prefix=categorical_cols)

    def encode_tags(self):
        """
        Transforma la columna 'tags', que contiene listas de etiquetas como strings, en variables binarias usando MultiLabelBinarizer.
        """
        def safe_literal_eval(val):
            try:
                return ast.literal_eval(val)
            except (ValueError, SyntaxError):
                return []
        self.df['tags'] = self.df['tags'].apply(safe_literal_eval)
        tags_encoded = pd.DataFrame(
            self.mlb_tags.fit_transform(self.df['tags']),
            columns=[f'tag_{tag}' for tag in self.mlb_tags.classes_],
            index=self.df.index
        )
        self.df = pd.concat([self.df.drop(columns=['tags']), tags_encoded], axis=1)

    def transform(self):
        """
        Ejecuta todos los pasos de ingeniería de características en orden.
        
        Retorna:
        - df (pd.DataFrame): DataFrame transformado listo para análisis o modelado.
        """
        self.drop_unnecessary_columns()
        self.add_price_diff_and_pictures_count()
        self.encode_booleans()
        self.add_title_features()
        self.add_date_features()
        self.encode_categories()
        self.encode_dummies()
        self.encode_tags()
        return self.df
