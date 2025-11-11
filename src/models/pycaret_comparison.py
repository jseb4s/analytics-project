import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


def run_pycaret_comparison(
    models_to_include: Optional[List[str]] = None,
    n_select: int = 5,
    optimize_metric: str = 'AUC',
    save_results: bool = True
) -> dict:
    """
    Ejecuta comparacion de modelos con PyCaret.
    
    Args:
        models_to_include: Lista de modelos a incluir (None = todos)
        n_select: Número de mejores modelos a seleccionar
        optimize_metric: Métrica para optimizar
        save_results: Si True, guarda resultados y figuras
        
    Returns:
        Diccionario con resultados y mejor modelo
    """
    try:
        from pycaret.classification import (
            setup, compare_models, tune_model, predict_model, 
            pull, save_model
        )
    except ImportError:
        logger.error("PyCaret no está instalado. Instala con: uv pip install pycaret")
        raise
    
    logger.info("\n" + "="*80)
    logger.info("INICIANDO ANALISIS CON PYCARET (AutoML)")
    logger.info("="*80)
    
    # cargar datos procesados
    logger.info("\n>>> Cargando datos procesados...")
    data_dir = Path("data/processed")
    
    X_train = pd.read_csv(data_dir / "X_train.csv")
    X_test = pd.read_csv(data_dir / "X_test.csv")
    y_train = pd.read_csv(data_dir / "y_train.csv").squeeze()
    y_test = pd.read_csv(data_dir / "y_test.csv").squeeze()
    
    # recombinar para PyCaret
    train_pycaret = X_train.copy()
    train_pycaret['churn'] = y_train.values
    
    test_pycaret = X_test.copy()
    test_pycaret['churn'] = y_test.values
    
    logger.info(f"Train set: {train_pycaret.shape}, Test set: {test_pycaret.shape}")
    
    # setup de PyCaret
    logger.info("\n>>> Configurando PyCaret...")
    logger.info(f"  Métrica de optimización: {optimize_metric}")
    
    train_pycaret = train_pycaret.reset_index(drop=True)
    test_pycaret = test_pycaret.reset_index(drop=True)

    clf = setup(
        data=train_pycaret,
        target='churn',
        session_id=123,
        normalize=False,  # ya normalizamos antes
        transformation=False,
        fold=5,
        test_data=test_pycaret,
        verbose=False,
        html=False,
        index=False
    )
    
    # modelos por defecto
    if models_to_include is None:
        models_to_include = ['svm', 'rf', 'ada', 'lr', 'gbc', 'lightgbm', 'dt', 'knn']
    
    logger.info(f"  Modelos a comparar: {models_to_include}")
    
    # comparar modelos
    logger.info(f"\n>>> Comparando {len(models_to_include)} modelos...")
    logger.info(f"  Seleccionando top {n_select} modelos")
    
    best_models = compare_models(
        include=models_to_include,
        sort=optimize_metric,
        n_select=n_select,
        verbose=False
    )
    
    # obtener resultados
    comparison_results = pull()
    
    print("\n" + "="*80)
    print("RESULTADOS DE COMPARACION - PYCARET")
    print("="*80)
    print(comparison_results)
    
    # tunear el mejor modelo
    logger.info(f"\n>>> Tuneando el mejor modelo...")
    best = best_models[0] if isinstance(best_models, list) else best_models
    
    tuned_best = tune_model(best, optimize=optimize_metric, verbose=False)
    
    logger.info("Modelo tuneado exitosamente")
    
    # evaluar en test
    logger.info("\n>>> Evaluando en conjunto de test...")
    predictions = predict_model(tuned_best, data=test_pycaret, verbose=False)
    
    # extraer metricas finales del modelo tuneado
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, 
        f1_score, roc_auc_score, confusion_matrix, roc_curve
    )
    
    y_true = predictions['churn']
    y_pred = predictions['prediction_label']
    y_pred_proba = predictions['prediction_score']
    
    final_metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1-Score': f1_score(y_true, y_pred),
        'ROC-AUC': roc_auc_score(y_true, y_pred_proba)
    }
    
    print("\n" + "="*80)
    print("METRICAS FINALES - MEJOR MODELO TUNEADO")
    print("="*80)
    for metric, value in final_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    results = {
        'comparison': comparison_results,
        'best_model': tuned_best,
        'predictions': predictions,
        'metrics': final_metrics
    }
    
    # guardar resultados
    if save_results:
        logger.info("\n>>> Guardando resultados...")
        
        # crear directorios
        results_dir = Path("reports/pycaret")
        figures_dir = Path("reports/figures/pycaret")
        models_dir = Path("trained_models/pycaret")
        results_dir.mkdir(parents=True, exist_ok=True)
        figures_dir.mkdir(parents=True, exist_ok=True)
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # guardar comparacion
        comparison_path = results_dir / "model_comparison.csv"
        comparison_results.to_csv(comparison_path, index=False)
        logger.info(f"Comparación guardada en: reports/pycaret/model_comparison.csv")
        
        # guardar metricas finales
        metrics_df = pd.DataFrame([final_metrics])
        metrics_path = results_dir / "final_metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)
        logger.info(f"Métricas finales guardadas en: reports/pycaret/final_metrics.csv")
        
        # guardar modelo
        model_path = models_dir / "best_model"
        save_model(tuned_best, str(model_path))
        logger.info(f"Modelo guardado en: trained_models/pycaret/best_model.pkl")
        
        # generar visualizaciones
        logger.info("\n>>> Generando visualizaciones...")
        
        # crear figura combinada: matriz de confusion + curva ROC
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # matriz de confusion (izquierda)
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                    xticklabels=['No Churn', 'Churn'],
                    yticklabels=['No Churn', 'Churn'])
        axes[0].set_title('Matriz de Confusion - PyCaret', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Valor Real')
        axes[0].set_xlabel('Prediccion')
        
        # agregar porcentajes
        for i in range(2):
            for j in range(2):
                percentage = cm[i, j] / cm[i].sum() * 100
                axes[0].text(j+0.5, i+0.7, f'({percentage:.1f}%)',
                            ha='center', va='center', fontsize=10, color='red')
        
        # curva ROC (derecha)
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc_score = final_metrics['ROC-AUC']
        
        axes[1].plot(fpr, tpr, color='darkorange', lw=2,
                    label=f'ROC curve (AUC = {auc_score:.3f})')
        axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        axes[1].set_xlim([0.0, 1.0])
        axes[1].set_ylim([0.0, 1.05])
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_title('Curva ROC - PyCaret', fontsize=14, fontweight='bold')
        axes[1].legend(loc="lower right")
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show() 
        # guardar figura combinada
        combined_path = figures_dir / "confusion_matrix_and_roc.png"
        plt.savefig(combined_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"  Matriz de confusion y ROC guardadas en: {combined_path}")
        
        # feature importance
        try:
            from pycaret.classification import plot_model
            plot_model(tuned_best, plot='feature', save=True, verbose=False)
            
            # mover archivo generado
            if Path("Feature Importance.png").exists():
                Path("Feature Importance.png").rename(figures_dir / "feature_importance.png")
                logger.info("  Feature importance generada")
        except Exception as e:
            logger.warning(f"  Error generando feature importance: {e}")
        
        logger.info(f"\nFiguras guardadas en: reports/figures/pycaret/")
    
    logger.info("\n" + "="*80)
    logger.info("ANALISIS CON PYCARET COMPLETADO")
    logger.info("="*80)
    
    return results


def main():
    """
    Función principal para ejecutar comparación con PyCaret.
    """
    # configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/pycaret_comparison.log', mode='w')
        ]
    )
    
    logger.info("="*80)
    logger.info("COMPARACION DE MODELOS CON PYCARET (AutoML)")
    logger.info("="*80)
    
    # verificar que existan datos procesados
    data_dir = Path("data/processed")
    required_files = ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]
    
    missing_files = [f for f in required_files if not (data_dir / f).exists()]
    
    if missing_files:
        logger.error(f"\nERROR: Faltan archivos procesados: {missing_files}")
        logger.error("Ejecuta primero: python main.py")
        return False
    
    # ejecutar comparacion
    try:
        results = run_pycaret_comparison(
            models_to_include=['svm', 'rf', 'ada', 'lr', 'gbc', 'lightgbm', 'dt', 'knn'],
            n_select=5,
            optimize_metric='AUC',
            save_results=True
        )
        
        logger.info("\nProceso completado exitosamente")
        logger.info("\nARCHIVOS GENERADOS:")
        logger.info("  - reports/pycaret/model_comparison.csv")
        logger.info("  - reports/pycaret/final_metrics.csv")
        logger.info("  - trained_models/pycaret/best_model.pkl")
        logger.info("  - reports/figures/pycaret/confusion_matrix_and_roc.png")
        logger.info("  - reports/figures/pycaret/feature_importance.png")
        
        return True
        
    except Exception as e:
        logger.error(f"\nError durante la ejecución: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    import sys
    
    success = main()
    
    if success:
        print("\nAnalisis con PyCaret completado exitosamente")
        print("Ver resultados en reports/pycaret/")
        sys.exit(0)
    else:
        print("\nAnalisis con PyCaret fallo - revisa los logs")
        sys.exit(1)