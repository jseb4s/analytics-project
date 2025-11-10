import logging
import numpy as np
from pathlib import Path
from typing import Optional

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from scikeras.wrappers import KerasClassifier

from .base_model import BaseModel, load_processed_data

logger = logging.getLogger(__name__)


def create_neural_network(input_dim: int,
                         units_layer1: int = 128,
                         units_layer2: int = 64,
                         dropout1: float = 0.3,
                         dropout2: float = 0.2,
                         learning_rate: float = 0.001) -> keras.Model:
    """
    Crea la arquitectura de la red neuronal.
    
    Args:
        input_dim: Dimension de entrada (numero de features)
        units_layer1: Neuronas en la primera capa oculta
        units_layer2: Neuronas en la segunda capa oculta
        dropout1: Dropout para la primera capa
        dropout2: Dropout para la segunda capa
        learning_rate: Tasa de aprendizaje
        
    Returns:
        Modelo Keras compilado
    """
    model = keras.Sequential([
        layers.Dense(units_layer1, activation="relu", 
                    input_dim=input_dim, 
                    name="hidden-dense-128-layer-1"),
        layers.Dropout(dropout1),
        layers.Dense(units_layer2, activation="relu", 
                    name="hidden-dense-64-layer-2"),
        layers.Dropout(dropout2),
        layers.Dense(1, activation="sigmoid", 
                    name="output-layer"),
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        loss="binary_crossentropy", 
        optimizer=optimizer, 
        metrics=["accuracy"]
    )
    
    return model


class NeuralNetworkModel(BaseModel):
    """Modelo de Red Neuronal."""
    
    def __init__(self, 
                 input_dim: int,
                 epochs: int = 30,
                 batch_size: int = 128,
                 validation_split: float = 0.2,
                 random_state: int = 42,
                 **nn_params):
        super().__init__("Neural Network", random_state)
        self.input_dim = input_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.nn_params = nn_params
        self.history = None
    
    def build_model(self) -> KerasClassifier:
        """Construye el modelo de red neuronal."""
        logger.info("Construyendo Red Neuronal")
        logger.info(f"  Input dim: {self.input_dim}")
        logger.info(f"  Epochs: {self.epochs}")
        logger.info(f"  Batch size: {self.batch_size}")
        
        return KerasClassifier(
            model=lambda: create_neural_network(self.input_dim, **self.nn_params),
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=self.validation_split,
            verbose=1
        )
    
    def train(self, X_train, y_train):
        """Entrena la red neuronal y guarda el historial."""
        super().train(X_train, y_train)
        # el historial se guarda automaticamente en el modelo KerasClassifier
        # pero podemos acceder a el si es necesario
    
    def plot_training_history(self, save_path: Optional[Path] = None):
        """
        Grafica el historial de entrenamiento (loss y accuracy).
        
        Args:
            save_path: Ruta donde guardar la figura (opcional)
        """
        if not hasattr(self.model, 'history_'):
            logger.warning("No hay historial de entrenamiento disponible")
            return
        
        import matplotlib.pyplot as plt
        
        history = self.model.history_
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss
        axes[0].plot(history['loss'], label='Train Loss')
        axes[0].plot(history['val_loss'], label='Validation Loss')
        axes[0].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Accuracy
        axes[1].plot(history['accuracy'], label='Train Accuracy')
        axes[1].plot(history['val_accuracy'], label='Validation Accuracy')
        axes[1].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            short_path = "/".join(save_path.parts[-3:])
            logger.info(f"Historial guardado en: {short_path}")
        
        plt.show()


def train_neural_network(save_model: bool = True,
                        save_figures: bool = True) -> NeuralNetworkModel:
    """
    Entrena modelo de Red Neuronal.
    
    Args:
        save_model: Si True, guarda el modelo entrenado
        save_figures: Si True, guarda las figuras
        
    Returns:
        Modelo entrenado
    """
    # cargar datos
    X_train, X_test, y_train, y_test = load_processed_data()
    
    # paths
    project_root = Path(__file__).parent.parent.parent
    model_path = project_root / "trained_models" / "neural_network" / "nn_model.pkl"
    results_path = project_root / "reports" / "figures" / "neural_network" / "nn_results.png"
    history_path = project_root / "reports" / "figures" / "neural_network" / "nn_training_history.png"
    
    # entrenar
    logger.info("\n" + "="*60)
    logger.info("ENTRENANDO RED NEURONAL")
    logger.info("="*60)
    
    nn_model = NeuralNetworkModel(
        input_dim=X_train.shape[1],
        epochs=30,
        batch_size=128,
        validation_split=0.2
    )
    
    nn_model.train(X_train, y_train)
    nn_model.evaluate(X_test, y_test)
    nn_model.print_metrics()
    
    # graficar resultados
    if save_figures:
        nn_model.plot_results(results_path)
        nn_model.plot_training_history(history_path)
    else:
        nn_model.plot_results()
        nn_model.plot_training_history()
    
    if save_model:
        nn_model.save_model(model_path)
    
    return nn_model


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    train_neural_network(save_model=True, save_figures=True)