import logging
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform

from .base_model import BaseModel, load_processed_data

logger = logging.getLogger(__name__)


class SVMModel(BaseModel):
    """Modelo SVM con configuracion baseline."""
    
    def __init__(self, random_state: int = 42, **svm_params):
        super().__init__("SVM Baseline", random_state)
        self.svm_params = svm_params
    
    def build_model(self) -> SVC:
        """Construye el modelo SVM baseline."""
        default_params = {
            'kernel': 'rbf',
            'C': 1.0,
            'gamma': 'scale',
            'class_weight': 'balanced',  # para datos desbalanceados
            'random_state': self.random_state,
            'probability': True
        }
        default_params.update(self.svm_params)
        
        logger.info(f"Construyendo SVM con parámetros: {default_params}")
        return SVC(**default_params)


class OptimizedSVMModel(BaseModel):
    """Modelo SVM optimizado con RandomizedSearchCV."""
    
    def __init__(self, 
                 random_state: int = 42,
                 n_iter: int = 50,
                 cv: int = 5,
                 n_jobs: int = -1):
        super().__init__("SVM Optimizado", random_state)
        self.n_iter = n_iter
        self.cv = cv
        self.n_jobs = n_jobs
        self.best_params = None
        self.cv_results = None
    
    def build_model(self) -> RandomizedSearchCV:
        """Construye el modelo SVM con busqueda de hiperparametros."""
        param_distributions = {
            'C': loguniform(0.01, 100),
            'gamma': loguniform(0.0001, 1),
            'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
            'degree': [2, 3, 4, 5],  # solo para poly
            'class_weight': ['balanced', None]
        }
        
        logger.info("Construyendo SVM con busqueda de hiperparametros")
        logger.info(f"  Iteraciones: {self.n_iter}")
        logger.info(f"  CV folds: {self.cv}")
        
        return RandomizedSearchCV(
            SVC(random_state=self.random_state, probability=True),
            param_distributions=param_distributions,
            n_iter=self.n_iter,
            cv=self.cv,
            scoring='roc_auc',
            n_jobs=self.n_jobs,
            verbose=1,
            random_state=self.random_state
        )
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Entrena el modelo y guarda los mejores parametros."""
        super().train(X_train, y_train)
        
        self.best_params = self.model.best_params_
        self.cv_results = pd.DataFrame(self.model.cv_results_)
        
        logger.info("\nMejores hiperparametros encontrados:")
        for param, value in self.best_params.items():
            logger.info(f"  {param}: {value}")
        logger.info(f"Mejor ROC-AUC (CV): {self.model.best_score_:.4f}")
    
    def print_best_params(self) -> None:
        """Imprime los mejores parametros encontrados."""
        print(f"\n{'='*50}")
        print("MEJORES HIPERPARAMETROS")
        print(f"{'='*50}")
        for param, value in self.best_params.items():
            print(f"  {param}: {value}")
        print(f"\nMejor ROC-AUC (CV): {self.model.best_score_:.4f}")
    
    def analyze_kernels(self) -> pd.DataFrame:
        """Analiza el desempeño por tipo de kernel."""
        if self.cv_results is None:
            raise ValueError("El modelo debe ser entrenado primero")
        
        kernel_analysis = self.cv_results.groupby('param_kernel').agg({
            'mean_test_score': ['mean', 'max'],
            'mean_fit_time': 'mean'
        }).round(4)
        
        kernel_analysis.columns = ['ROC-AUC Promedio', 'ROC-AUC Máximo', 'Tiempo (s)']
        
        print(f"\n{'='*50}")
        print("ANALISIS DE RENDIMIENTO POR KERNEL")
        print(f"{'='*50}")
        print(kernel_analysis)
        
        return kernel_analysis


def train_svm_models(save_models: bool = True,
                    save_figures: bool = True) -> Dict[str, BaseModel]:
    """
    Entrena modelos SVM baseline y optimizado.
    
    Args:
        save_models: Si True, guarda los modelos entrenados
        save_figures: Si True, guarda las figuras
        
    Returns:
        Diccionario con los modelos entrenados
    """
    # Cargar datos
    X_train, X_test, y_train, y_test = load_processed_data()
    
    # Paths para guardar
    project_root = Path(__file__).parent.parent.parent
    models_dir = project_root / "trained_models" / "svm"
    figures_dir = project_root / "reports" / "figures" / "svm"
    
    results = {}
    
    # 1.SVM Baseline
    logger.info("\n" + "="*60)
    logger.info("ENTRENANDO SVM BASELINE")
    logger.info("="*60)
    
    svm_baseline = SVMModel()
    svm_baseline.train(X_train, y_train)
    svm_baseline.evaluate(X_test, y_test)
    svm_baseline.print_metrics()
    
    if save_figures:
        svm_baseline.plot_results(figures_dir / "svm_baseline_results.png")
    else:
        svm_baseline.plot_results()
    
    if save_models:
        svm_baseline.save_model(models_dir / "svm_baseline.pkl")
    
    results['baseline'] = svm_baseline
    
    # 2.SVM Optimizado
    logger.info("\n" + "="*60)
    logger.info("ENTRENANDO SVM OPTIMIZADO")
    logger.info("="*60)
    
    svm_optimized = OptimizedSVMModel(n_iter=50, cv=5)
    svm_optimized.train(X_train, y_train)
    svm_optimized.print_best_params()
    svm_optimized.analyze_kernels()
    
    svm_optimized.evaluate(X_test, y_test)
    svm_optimized.print_metrics()
    
    if save_figures:
        svm_optimized.plot_results(figures_dir / "svm_optimized_results.png")
    else:
        svm_optimized.plot_results()
    
    if save_models:
        svm_optimized.save_model(models_dir / "svm_optimized.pkl")
    
    results['optimized'] = svm_optimized
    
    # 3.Comparacion
    comparison = pd.DataFrame({
        'Métrica': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
        'SVM Baseline': [
            svm_baseline.metrics['accuracy'],
            svm_baseline.metrics['precision'],
            svm_baseline.metrics['recall'],
            svm_baseline.metrics['f1_score'],
            svm_baseline.metrics['roc_auc']
        ],
        'SVM Optimizado': [
            svm_optimized.metrics['accuracy'],
            svm_optimized.metrics['precision'],
            svm_optimized.metrics['recall'],
            svm_optimized.metrics['f1_score'],
            svm_optimized.metrics['roc_auc']
        ]
    })
    
    comparison['Mejora (%)'] = (
        (comparison['SVM Optimizado'] - comparison['SVM Baseline']) / 
        comparison['SVM Baseline'] * 100
    ).round(2)
    
    print(f"\n{'='*60}")
    print("COMPARACIÓN: BASELINE vs OPTIMIZADO")
    print(f"{'='*60}")
    print(comparison.round(4))
    
    results['comparison'] = comparison
    
    return results


if __name__ == "__main__":
    # configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # entrenar modelos
    results = train_svm_models(save_models=True, save_figures=True)