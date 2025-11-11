import logging
from pathlib import Path
from typing import Optional

from utils.logger_config import setup_logger
from utils.data_loader import load_bank_churn_data
from preprocessing.pipeline import BankChurnPreprocessor, save_processed_data
from models.pycaret_comparison import run_pycaret_comparison


def run_preprocessing(categorical_cols: Optional[list] = None) -> bool:
    """Ejecuta el preprocesamiento de datos."""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("\n" + "="*80)
        logger.info("PREPROCESAMIENTO DE DATOS")
        logger.info("="*80)
        
        logger.info("\n>>> Cargando datos raw...")
        df = load_bank_churn_data()
        
        if categorical_cols is None:
            categorical_cols = [
                'gender', 'education_level', 'marital_status', 
                'income_category', 'card_category'
            ]
        
        logger.info(f"Columnas categoricas: {categorical_cols}")
        
        logger.info("\n>>> Ejecutando pipeline de preprocesamiento...")
        preprocessor = BankChurnPreprocessor(categorical_cols=categorical_cols)
        X_train, X_test, y_train, y_test = preprocessor.fit_transform(df)
        
        logger.info("\n>>> Guardando datos procesados...")
        save_processed_data(X_train, X_test, y_train, y_test)
        
        logger.info("\n>>> Guardando scaler...")
        scaler_path = "trained_models/preprocessing/scaler.pkl"
        preprocessor.save_scaler(scaler_path)
        
        logger.info("\nPREPROCESAMIENTO COMPLETADO EXITOSAMENTE")
        return True
        
    except Exception as e:
        logger.error(f"\nERROR EN PREPROCESAMIENTO: {e}", exc_info=True)
        return False


def main():
    """Funcion principal que ejecuta el pipeline de PyCaret."""
    # configurar logging
    log_file = "logs/pycaret_pipeline.log"
    setup_logger(name=__name__, level=logging.INFO, log_file=log_file, mode='w')
    
    logger = logging.getLogger(__name__)
    
    logger.info("\n" + "="*80)
    logger.info("PIPELINE PYCARET - AUTOML")
    logger.info("="*80)
    
    # verificar datos raw
    raw_data_path = Path("data/raw/bank_churn.xlsx")
    if not raw_data_path.exists():
        logger.error(f"\nERROR: No se encontro el archivo de datos en {raw_data_path}")
        logger.error("Por favor, coloca el archivo bank_churn.xlsx en data/raw/")
        return False
    
    # paso 1: preprocesamiento
    data_dir = Path("data/processed")
    required_files = ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]
    missing_files = [f for f in required_files if not (data_dir / f).exists()]
    
    if missing_files:
        logger.info("\n>>> Datos procesados no encontrados. Ejecutando preprocesamiento...")
        preprocessing_success = run_preprocessing()
        
        if not preprocessing_success:
            logger.error("\nPipeline interrumpido: error en preprocesamiento")
            return False
    else:
        logger.info("\n>>> Datos procesados encontrados. Omitiendo preprocesamiento...")
    
    # paso 2: analisis con PyCaret
    try:
        logger.info("\n" + "="*80)
        logger.info("ANALISIS CON PYCARET")
        logger.info("="*80)
        
        results = run_pycaret_comparison(
            models_to_include=['svm', 'rf', 'ada', 'lr', 'gbc', 'lightgbm', 'dt', 'knn'],
            n_select=5,
            optimize_metric='AUC',
            save_results=True
        )
        
        # mostrar resumen
        if 'metrics' in results:
            logger.info("\n" + "="*80)
            logger.info("RESUMEN FINAL")
            logger.info("="*80)
            for metric, value in results['metrics'].items():
                logger.info(f"  {metric}: {value:.4f}")
        
    except Exception as e:
        logger.error(f"\nERROR EN ANALISIS PYCARET: {e}", exc_info=True)
        return False
    
    # pipeline completado
    logger.info("\n" + "="*80)
    logger.info("PIPELINE PYCARET FINALIZADO EXITOSAMENTE")
    logger.info("="*80)
    
    logger.info("\nARCHIVOS GENERADOS:")
    logger.info("   Datos procesados: data/processed/")
    logger.info("   Modelo: trained_models/pycaret/")
    logger.info("   Resultados: reports/pycaret/")
    logger.info("   Figuras: reports/figures/pycaret/")
    logger.info(f"   Log completo: {log_file}")
    
    return True


if __name__ == "__main__":
    import sys
    
    success = main()
    
    if success:
        print("\nPipeline PyCaret ejecutado exitosamente")
        print("Ver resultados en reports/pycaret/")
        sys.exit(0)
    else:
        print("\nPipeline PyCaret fallo - revisa los logs")
        sys.exit(1)