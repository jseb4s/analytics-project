import logging
from pathlib import Path
from typing import Optional

from utils.data_loader import load_bank_churn_data
from preprocessing.pipeline import BankChurnPreprocessor, save_processed_data
from models.train import train_all_models


def setup_logging(log_file: Optional[str] = None) -> None:
    """
    Configura el sistema de logging.
    
    Args:
        log_file: Ruta al archivo de log (opcional)
    """
    handlers = [logging.StreamHandler()]
    
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def run_preprocessing(categorical_cols: Optional[list] = None) -> bool:
    """
    Ejecuta el preprocesamiento de datos.
    
    Args:
        categorical_cols: Lista de columnas categoricas
        
    Returns:
        True si el preprocesamiento fue exitoso
    """
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("\n" + "="*80)
        logger.info("PREPROCESAMIENTO DE DATOS")
        logger.info("="*80)
        
        # Cargar datos raw
        logger.info("\n>>> Cargando datos raw...")
        df = load_bank_churn_data()
        
        # Definir columnas categóricas por defecto
        if categorical_cols is None:
            categorical_cols = [
                'gender', 'education_level', 'marital_status', 
                'income_category', 'card_category'
            ]
        
        logger.info(f"Columnas categóricas: {categorical_cols}")
        
        # Preprocesar
        logger.info("\n>>> Ejecutando pipeline de preprocesamiento...")
        preprocessor = BankChurnPreprocessor(categorical_cols=categorical_cols)
        X_train, X_test, y_train, y_test = preprocessor.fit_transform(df)
        
        # Guardar datos procesados
        logger.info("\n>>> Guardando datos procesados...")
        save_processed_data(X_train, X_test, y_train, y_test)
        
        # Guardar scaler
        logger.info("\n>>> Guardando scaler...")
        scaler_path = Path("trained_models/preprocessing/scaler.pkl")
        preprocessor.save_scaler(str(scaler_path))
        
        logger.info("\nPREPROCESAMIENTO COMPLETADO EXITOSAMENTE")
        return True
        
    except Exception as e:
        logger.error(f"\nERROR EN PREPROCESAMIENTO: {e}", exc_info=True)
        return False


def run_training(save_models: bool = True, save_figures: bool = True) -> bool:
    """
    Ejecuta el entrenamiento de todos los modelos.
    
    Args:
        save_models: Si True, guarda los modelos entrenados
        save_figures: Si True, guarda las figuras
        
    Returns:
        True si el entrenamiento fue exitoso
    """
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("\n" + "="*80)
        logger.info("ENTRENAMIENTO DE MODELOS")
        logger.info("="*80)
        
        # entrenar todos los modelos
        results = train_all_models(save_models=save_models, save_figures=save_figures)
        
        # mostrar resumen final
        if 'comparison' in results:
            logger.info("\n" + "="*80)
            logger.info("RESUMEN FINAL - COMPARACION DE MODELOS")
            logger.info("="*80)
            print("\n" + str(results['comparison'].to_string(index=False)))
            
            # Mejor modelo
            best_model = results['comparison'].iloc[0]
            logger.info(f"\nMEJOR MODELO: {best_model['Modelo']}")
            logger.info(f"   ROC-AUC: {best_model['ROC-AUC']:.4f}")
        
        logger.info("\nENTRENAMIENTO COMPLETADO EXITOSAMENTE")
        return True
        
    except Exception as e:
        logger.error(f"\nERROR EN ENTRENAMIENTO: {e}", exc_info=True)
        return False


def main():
    """
    Funcipn principal que ejecuta el pipeline completo.
    """
    # logging
    log_file = "logs/main_pipeline.log"
    setup_logging(log_file)
    
    logger = logging.getLogger(__name__)
    
    logger.info("\n" + "="*80)
    logger.info("INICIANDO PIPELINE COMPLETO DE MACHINE LEARNING")
    logger.info("="*80)
    
    # verificar que existan los datos raw
    raw_data_path = Path("data/raw/bank_churn.xlsx")
    if not raw_data_path.exists():
        logger.error(f"\nERROR: No se encontro el archivo de datos en {raw_data_path}")
        logger.error("Por favor, coloca el archivo bank_churn.xlsx en data/raw/")
        return False
    
    # paso 1: preprocesamiento
    preprocessing_success = run_preprocessing()
    
    if not preprocessing_success:
        logger.error("\nPipeline interrumpido: error en preprocesamiento")
        return False
    
    #paso 2: entrenamiento
    training_success = run_training(save_models=True, save_figures=True)
    
    if not training_success:
        logger.error("\nPipeline interrumpido: error en entrenamiento")
        return False
    
    # pipeline completado
    logger.info("\n" + "="*80)
    logger.info("PIPELINE COMPLETO FINALIZADO EXITOSAMENTE")
    logger.info("="*80)
    
    # informacion de archivos generados
    logger.info("\nARCHIVOS GENERADOS:")
    logger.info("   Datos procesados: data/processed/")
    logger.info("   Modelos entrenados: models/")
    logger.info("   Figuras: reports/figures/")
    logger.info("   Comparación: models/model_comparison.csv")
    logger.info(f"   Log completo: {log_file}")
    
    return True


if __name__ == "__main__":
    import sys
    
    success = main()
    
    if success:
        print("\nPipeline ejecutado exitosamente")
        print("Ver resultados en models/model_comparison.csv")
        sys.exit(0)
    else:
        print("\nPipeline fallo - revisa los logs para más detalles")
        sys.exit(1)