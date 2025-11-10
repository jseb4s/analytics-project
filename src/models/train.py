import logging
import pandas as pd
from pathlib import Path
from typing import Dict

from .svm_model import train_svm_models
from .random_forest_model import train_random_forest
from .ligthgbm_model import train_lightgbm
from .gradient_boosting_model import train_gradient_boosting
from .neural_network_model import train_neural_network

logger = logging.getLogger(__name__)


def train_all_models(save_models: bool = True,
                    save_figures: bool = True) -> Dict:
    """
    Entrena todos los modelos y genera comparacion.
    
    Args:
        save_models: Si True, guarda los modelos entrenados
        save_figures: Si True, guarda las figuras
        
    Returns:
        Diccionario con todos los resultados
    """
    logger.info("\n" + "="*80)
    logger.info("INICIANDO ENTRENAMIENTO DE TODOS LOS MODELOS")
    logger.info("="*80)
    
    results = {}
    
    # 1. SVM (Baseline y Optimizado)
    try:
        logger.info("\n>>> Entrenando modelos SVM...")
        svm_results = train_svm_models(save_models, save_figures)
        results['svm_baseline'] = svm_results['baseline']
        results['svm_optimized'] = svm_results['optimized']
    except Exception as e:
        logger.error(f"Error entrenando SVM: {e}")
    
    # 2. Random Forest
    try:
        logger.info("\n>>> Entrenando Random Forest...")
        results['random_forest'] = train_random_forest(save_models, save_figures)
    except Exception as e:
        logger.error(f"Error entrenando Random Forest: {e}")
    
    # 3. LightGBM
    try:
        logger.info("\n>>> Entrenando LightGBM...")
        results['lightgbm'] = train_lightgbm(save_models, save_figures)
    except Exception as e:
        logger.error(f"Error entrenando LightGBM: {e}")
    
    # 4. Gradient Boosting
    try:
        logger.info("\n>>> Entrenando Gradient Boosting...")
        results['gradient_boosting'] = train_gradient_boosting(save_models, save_figures)
    except Exception as e:
        logger.error(f"Error entrenando Gradient Boosting: {e}")
    
    # 5. Neural Network
    try:
        logger.info("\n>>> Entrenando Red Neuronal...")
        results['neural_network'] = train_neural_network(save_models, save_figures)
    except Exception as e:
        logger.error(f"Error entrenando Red Neuronal: {e}")
    
    # 6. Comparación Final
    logger.info("\n" + "="*80)
    logger.info("COMPARACION FINAL DE TODOS LOS MODELOS")
    logger.info("="*80)
    
    comparison_data = []
    for model_name, model in results.items():
        if hasattr(model, 'metrics') and model.metrics:
            comparison_data.append({
                'Modelo': model.model_name,
                'Accuracy': model.metrics['accuracy'],
                'Precision': model.metrics['precision'],
                'Recall': model.metrics['recall'],
                'F1-Score': model.metrics['f1_score'],
                'ROC-AUC': model.metrics['roc_auc']
            })
    
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('ROC-AUC', ascending=False)
        
        print("\n")
        print(comparison_df.round(4).to_string(index=False))
        
        # Guardar comparacion
        if save_models:
            project_root = Path(__file__).parent.parent.parent
            comparison_path = project_root / "trained_models" / "model_comparison.csv"
            comparison_path.parent.mkdir(parents=True, exist_ok=True)
            comparison_df.to_csv(comparison_path, index=False)
            short_c_path = "/".join(comparison_path.parts[-2:])
            logger.info(f"\nComparacion guardada en: {short_c_path}")
        
        results['comparison'] = comparison_df
    
    logger.info("\n" + "="*80)
    logger.info("ENTRENAMIENTO COMPLETADO")
    logger.info("="*80)
    
    return results


if __name__ == "__main__":
    # configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('training.log')
        ]
    )
    
    # entrenar todos los modelos
    results = train_all_models(save_models=True, save_figures=True)
    
    # mostrar el mejor modelo
    if 'comparison' in results:
        best_model = results['comparison'].iloc[0]
        print(f"\nMEJOR MODELO: {best_model['Modelo']}")
        print(f"   ROC-AUC: {best_model['ROC-AUC']:.4f}")