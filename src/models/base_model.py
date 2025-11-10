import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from abc import ABC, abstractmethod

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """
    Clase base abstracta para todos los modelos.
    Define la interfaz comun y metodos compartidos.
    """
    
    def __init__(self, model_name: str, random_state: int = 42):
        """
        Inicializa el modelo base.
        
        Args:
            model_name: Nombre descriptivo del modelo
            random_state: Semilla para reproducibilidad
        """
        self.model_name = model_name
        self.random_state = random_state
        self.model = None
        self.metrics = {}
        self.predictions = {}
        
    @abstractmethod
    def build_model(self) -> Any:
        """
        Construye y retorna el modelo.
        Debe ser implementado por cada subclase.
        """
        pass
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Entrena el modelo.
        
        Args:
            X_train: Features de entrenamiento
            y_train: Target de entrenamiento
        """
        if self.model is None:
            self.model = self.build_model()
        
        logger.info(f"Entrenando {self.model_name}...")
        self.model.fit(X_train, y_train)
        logger.info(f"{self.model_name} entrenado exitosamente")
    
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Realiza predicciones.
        
        Args:
            X_test: Features de test
            
        Returns:
            Predicciones (0 o 1)
        """
        if self.model is None:
            raise ValueError("El modelo no ha sido entrenado")
        
        return self.model.predict(X_test)
    
    def predict_proba(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Calcula probabilidades de prediccion.
        
        Args:
            X_test: Features de test
            
        Returns:
            Probabilidades de la clase positiva
        """
        if self.model is None:
            raise ValueError("El modelo no ha sido entrenado")
        
        return self.model.predict_proba(X_test)[:, 1]
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evalua el modelo y calcula metricas.
        
        Args:
            X_test: Features de test
            y_test: Target de test
            
        Returns:
            Diccionario con metricas de desempeño
        """
        logger.info(f"Evaluando {self.model_name}...")
        
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)
        
        # guardar predicciones
        self.predictions = {
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'y_test': y_test
        }
        
        # calcular metricas
        self.metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        logger.info(f"Metricas de {self.model_name}:")
        for metric, value in self.metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return self.metrics
    
    def print_metrics(self) -> None:
        """Imprime las metricas de desempeño."""
        print(f"\n{'='*50}")
        print(f"{self.model_name.upper()} - METRICAS DE DESEMPEÑO")
        print(f"{'='*50}")
        print(f"Accuracy:  {self.metrics['accuracy']:.4f}")
        print(f"Precision: {self.metrics['precision']:.4f}")
        print(f"Recall:    {self.metrics['recall']:.4f}")
        print(f"F1-Score:  {self.metrics['f1_score']:.4f}")
        print(f"ROC-AUC:   {self.metrics['roc_auc']:.4f}")
        
        print(f"\n{'-'*50}")
        print("REPORTE DE CLASIFICACION:")
        print(f"{'-'*50}")
        print(classification_report(
            self.predictions['y_test'],
            self.predictions['y_pred'],
            target_names=['No Churn', 'Churn']
        ))
    
    def plot_results(self, save_path: Optional[Path] = None) -> None:
        """
        Genera visualizaciones de resultados.
        
        Args:
            save_path: Ruta donde guardar las figuras (opcional)
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # matriz de confusion
        cm = confusion_matrix(self.predictions['y_test'], self.predictions['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                    xticklabels=['No Churn', 'Churn'],
                    yticklabels=['No Churn', 'Churn'])
        axes[0].set_title(f'Matriz de confusion - {self.model_name}',
                         fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Valor real')
        axes[0].set_xlabel('Prediccion')
        
        # agregar porcentajes
        for i in range(2):
            for j in range(2):
                percentage = cm[i, j] / cm[i].sum() * 100
                axes[0].text(j+0.5, i+0.7, f'({percentage:.1f}%)',
                            ha='center', va='center', fontsize=10, color='red')
        
        # curva roc
        fpr, tpr, _ = roc_curve(self.predictions['y_test'],
                                self.predictions['y_pred_proba'])
        auc_score = self.metrics['roc_auc']
        
        axes[1].plot(fpr, tpr, color='darkorange', lw=2,
                    label=f'ROC curve (AUC = {auc_score:.3f})')
        axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                    label='Random')
        axes[1].set_xlim([0.0, 1.0])
        axes[1].set_ylim([0.0, 1.05])
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_title(f'Curva ROC - {self.model_name}',
                         fontsize=14, fontweight='bold')
        axes[1].legend(loc="lower right")
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            short_path = "/".join(save_path.parts[-3:])
            logger.info(f"Figura guardada en: {short_path}")
        
        plt.show()
    
    def save_model(self, filepath: Path) -> None:
        """
        Guarda el modelo entrenado.
        
        Args:
            filepath: Ruta donde guardar el modelo
        """
        if self.model is None:
            raise ValueError("No hay modelo para guardar")
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, filepath)
        short_path = "/".join(filepath.parts[-3:])
        logger.info(f"Modelo guardado en: {short_path}")
    
    def load_model(self, filepath: Path) -> None:
        """
        Carga un modelo previamente guardado.
        
        Args:
            filepath: Ruta del modelo guardado
        """
        self.model = joblib.load(filepath)
        short_path = "/".join(filepath.parts[-3:])
        logger.info(f"Modelo cargado desde: {short_path}")


def load_processed_data(data_dir: Optional[Path] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Carga los datos procesados.
    
    Args:
        data_dir: Directorio de datos procesados (default: data/processed/)
        
    Returns:
        Tupla (X_train, X_test, y_train, y_test)
    """
    if data_dir is None:
        data_dir = Path("data/processed")
    
    short_path = "/".join(data_dir.parts[-2:])
    logger.info(f"Cargando datos procesados desde: {short_path}")
    
    X_train = pd.read_csv(data_dir / "X_train.csv")
    X_test = pd.read_csv(data_dir / "X_test.csv")
    y_train = pd.read_csv(data_dir / "y_train.csv").squeeze()
    y_test = pd.read_csv(data_dir / "y_test.csv").squeeze()
    
    logger.info(f"Datos cargados - train: {X_train.shape}, Test: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test