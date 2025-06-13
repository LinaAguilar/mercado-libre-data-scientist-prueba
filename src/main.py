from data_analyzer import DataAnalyzer

def main():
    # Ruta al archivo CSV
    ruta_archivo = 'data/new_items_dataset.csv'

    # Crear una instancia del analizador de datos
    analyzer = DataAnalyzer(ruta_archivo)

    # Leer y filtrar los datos
    analyzer.read_data()

    # Mostrar resumen estadístico
    analyzer.summary_statistics()

    # Manejar valores faltantes o inconsistentes
    analyzer.handle_missing_values()

    # Detectar outliers en el precio de los productos
    analyzer.detect_price_outliers()

if __name__ == '__main__':
    main()
