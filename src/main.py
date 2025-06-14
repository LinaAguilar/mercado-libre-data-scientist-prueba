from data_analyzer import DataAnalyzer
from feature_engineering import FeatureEngineer
from price_predictor import PricePredictor
from sales_predictor import SalesPredictor

def main():
    # Ruta al archivo CSV
    ruta_archivo = 'data/new_items_dataset.csv'

    analyzer = DataAnalyzer(ruta_archivo)
    df_procesado = analyzer.process() 

    print("EDA realizado con éxito, resultando en un DF de", len(df_procesado), "filas.")

    engineer = FeatureEngineer(df_procesado)
    df_final = engineer.transform()

    print("Feature engineering realizado con éxito", len(df_final))

    predictor = PricePredictor(df_final)
    predictor.remove_outliers()
    predictor.train_test_split()
    predictor.train_models()
    predictor.cross_validate(predictor.trained_models['RandomForest'], 'RandomForest')
    predictor.cross_validate(predictor.trained_models['XGBoost'], 'XGBoost')
    best_model_name, best_metrics = predictor.compare_models()
    # Para predecir nuevos precios:
    # y_pred = predictor.predict(X_nuevo)

    sales_predictor = SalesPredictor(df_final)
    
    sales_predictor.prepare_data()
    sales_predictor.split_data(use_smote=True)
    sales_predictor.train_models()
    sales_predictor.evaluate()
    sales_predictor.optimize_threshold()
    sales_predictor.cross_validate_with_threshold_tuning()

if __name__ == '__main__':
    main()
