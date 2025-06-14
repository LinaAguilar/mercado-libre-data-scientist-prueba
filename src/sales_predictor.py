import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score
)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

class SalesPredictor:
    """
    Clase para predecir si un producto será vendido (`was_sold`) usando modelos de clasificación.

    Permite preparar los datos, balancear la clase con SMOTE, entrenar modelos, evaluar con métricas 
    de clasificación, optimizar el umbral de decisión y realizar validación cruzada con ajuste de umbral.

    Atributos:
        df (pd.DataFrame): DataFrame con los datos de entrada.
        models (dict): Diccionario para almacenar modelos entrenados.
        X (pd.DataFrame): Variables predictoras.
        y (pd.Series): Variable objetivo binaria (0: no vendido, 1: vendido).
    """

    def __init__(self, df):
        """
        Inicializa el predictor con los datos originales.

        Args:
            df (pd.DataFrame): DataFrame con columnas como 'id', 'sold_quantity', etc.
        """
        self.df = df.copy()
        self.models = {}

    def prepare_data(self):
        """
        Crea la variable objetivo binaria 'was_sold' a partir de 'sold_quantity',
        y separa las variables predictoras de la variable objetivo.
        También grafica la distribución de clases.
        """
        self.df['was_sold'] = (self.df['sold_quantity'] > 0).astype(int)

        # Visualización
        self.df['was_sold'].value_counts(normalize=True).plot(kind='bar')
        plt.title("Distribución de la variable objetivo")
        plt.xticks(ticks=[0, 1], labels=["No vendido", "Vendido"], rotation=0)
        plt.ylabel("Proporción")
        plt.show()

        self.X = self.df.drop(columns=['id', 'sold_quantity', 'was_sold'])
        self.y = self.df['was_sold']

    def split_data(self, use_smote=False):
        """
        Divide los datos en entrenamiento y prueba (80/20), con opción de aplicar SMOTE
        para balancear las clases en el conjunto de entrenamiento.

        Args:
            use_smote (bool): Si True, aplica SMOTE al conjunto de entrenamiento.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, stratify=self.y, random_state=42
        )

        if use_smote:
            smote = SMOTE(random_state=42)
            self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)
            print("Distribución después de SMOTE:", Counter(self.y_train))
        else:
            print("Distribución sin SMOTE:", Counter(self.y_train))

    def train_models(self):
        """
        Entrena modelos de clasificación usando Random Forest y XGBoost.
        Se usa `class_weight='balanced'` en RandomForest y `scale_pos_weight` en XGBoost
        para abordar el desbalanceo de clases.
        """
        spw = (self.y_train == 0).sum() / (self.y_train == 1).sum()

        self.models['RandomForest'] = RandomForestClassifier(class_weight='balanced', random_state=42)
        self.models['XGBoost'] = XGBClassifier(
            scale_pos_weight=spw, use_label_encoder=False, eval_metric='logloss', random_state=42
        )

        for name, model in self.models.items():
            model.fit(self.X_train, self.y_train)

    def evaluate(self):
        """
        Evalúa todos los modelos entrenados sobre el conjunto de prueba.
        Muestra el classification report y la matriz de confusión para cada uno.
        """
        for name, model in self.models.items():
            y_pred = model.predict(self.X_test)
            print(f"\nEvaluación de {name}:")
            print(classification_report(self.y_test, y_pred))
            print("Matriz de confusión:")
            print(confusion_matrix(self.y_test, y_pred))

    def optimize_threshold(self, model_name='XGBoost'):
        """
        Ajusta el umbral de probabilidad para maximizar el F1-score usando el modelo indicado.

        Args:
            model_name (str): Nombre del modelo a usar (por defecto 'XGBoost').
        """
        model = self.models[model_name]
        y_proba = model.predict_proba(self.X_test)[:, 1]

        best_f1, best_thresh = 0, 0.5
        for thresh in np.arange(0.1, 0.9, 0.05):
            y_pred = (y_proba >= thresh).astype(int)
            f1 = f1_score(self.y_test, y_pred)
            if f1 > best_f1:
                best_f1, best_thresh = f1, thresh

        print(f"Mejor threshold: {best_thresh:.2f} con F1-score: {best_f1:.4f}")

        y_pred_best = (y_proba >= best_thresh).astype(int)
        print(classification_report(self.y_test, y_pred_best))
        print("Matriz de confusión:")
        print(confusion_matrix(self.y_test, y_pred_best))

    def cross_validate_with_threshold_tuning(self):
        """
        Realiza validación cruzada estratificada (5-fold) y, en cada fold,
        entrena un modelo XGBoost, predice probabilidades y ajusta el mejor threshold
        que maximiza F1-score. Reporta los promedios de F1, precisión y recall.
        """
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        thresholds = np.arange(0.1, 0.9, 0.05)
        metrics = []

        for train_idx, val_idx in cv.split(self.X, self.y):
            X_train_cv, X_val_cv = self.X.iloc[train_idx], self.X.iloc[val_idx]
            y_train_cv, y_val_cv = self.y.iloc[train_idx], self.y.iloc[val_idx]

            spw = (y_train_cv == 0).sum() / (y_train_cv == 1).sum()
            model = XGBClassifier(
                scale_pos_weight=spw, random_state=42,
                use_label_encoder=False, eval_metric='logloss'
            )
            model.fit(X_train_cv, y_train_cv)
            y_proba = model.predict_proba(X_val_cv)[:, 1]

            best_f1 = 0
            best_metrics = {}

            for t in thresholds:
                y_pred = (y_proba >= t).astype(int)
                f1 = f1_score(y_val_cv, y_pred)
                if f1 > best_f1:
                    best_metrics = {
                        'threshold': t,
                        'f1': f1,
                        'precision': precision_score(y_val_cv, y_pred),
                        'recall': recall_score(y_val_cv, y_pred),
                        'confusion_matrix': confusion_matrix(y_val_cv, y_pred),
                        'report': classification_report(y_val_cv, y_pred, output_dict=True)
                    }
                    best_f1 = f1

            metrics.append(best_metrics)

        avg_f1 = np.mean([m['f1'] for m in metrics])
        avg_precision = np.mean([m['precision'] for m in metrics])
        avg_recall = np.mean([m['recall'] for m in metrics])

        print(f"\nPromedios en CV con tuning:")
        print(f"F1-score: {avg_f1:.4f}, Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}")
