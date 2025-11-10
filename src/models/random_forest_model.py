import logging
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

from .base_model import BaseModel, load_processed_data

logger = logging.getLogger(__name__)


class RandomForestModel(BaseModel):
    """Modelo Random Forest Classifier."""
    
    def __init__(self, random_state: int = 42, **rf_params):
        super().__init__("Random Forest", random_state)
        self.rf_params = rf_params
    
    def build_model(self) -> RandomForestClassifier:
        """Construye el modelo Random Forest."""
        default_params = {
            'n_estimators': 100,
            'random_state': self.random_state,
            'n_jobs': -1
        }
        default_params.update(self.rf_params)
        
        logger.info(f"Construyendo Random Forest con parametros: {default_params}")
        return RandomForestClassifier(**default_params)


def train_random_forest(save_model: bool = True,
                       save_figure: bool = True) -> RandomForestModel:
    """
    Entrena modelo Random Forest.
    
    Args:
        save_model: Si True, guarda el modelo entrenado
        save_figure: Si True, guarda la figura
        
    Returns:
        Modelo entrenado
    """
    # cargar datos
    X_train, X_test, y_train, y_test = load_processed_data()
    
    # paths
    project_root = Path(__file__).parent.parent.parent
    model_path = project_root / "trained_models" / "random_forest" / "rf_model.pkl"
    figure_path = project_root / "reports" / "figures" / "random_forest" / "rf_results.png"
    
    # entrenar
    logger.info("\n" + "="*60)
    logger.info("ENTRENANDO RANDOM FOREST")
    logger.info("="*60)
    
    rf_model = RandomForestModel()
    rf_model.train(X_train, y_train)
    rf_model.evaluate(X_test, y_test)
    rf_model.print_metrics()
    
    if save_figure:
        rf_model.plot_results(figure_path)
    else:
        rf_model.plot_results()
    
    if save_model:
        rf_model.save_model(model_path)
    
    return rf_model


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    train_random_forest(save_model=True, save_figure=True)