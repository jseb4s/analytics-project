import logging
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier

from .base_model import BaseModel, load_processed_data

logger = logging.getLogger(__name__)


class GradientBoostingModel(BaseModel):
    """Modelo Gradient Boosting Classifier."""
    
    def __init__(self, random_state: int = 42, **gb_params):
        super().__init__("Gradient Boosting", random_state)
        self.gb_params = gb_params
    
    def build_model(self) -> GradientBoostingClassifier:
        """Construye el modelo Gradient Boosting."""
        default_params = {
            'n_estimators': 200,
            'learning_rate': 0.1,
            'max_depth': 3,
            'subsample': 0.8,
            'random_state': self.random_state
        }
        default_params.update(self.gb_params)
        
        logger.info(f"Construyendo Gradient Boosting con parametros: {default_params}")
        return GradientBoostingClassifier(**default_params)


def train_gradient_boosting(save_model: bool = True,
                           save_figure: bool = True) -> GradientBoostingModel:
    """
    Entrena modelo Gradient Boosting.
    
    Args:
        save_model: Si True, guarda el modelo entrenado
        save_figure: Si True, guarda la figura
        
    Returns:
        Modelo entrenado
    """
    X_train, X_test, y_train, y_test = load_processed_data()
    
    model_path = Path("trained_models/gradient_boosting/gb_model.pkl")
    figure_path = Path("reports/figures/gradient_boosting/gb_results.png")
    
    logger.info("\n" + "="*60)
    logger.info("ENTRENANDO GRADIENT BOOSTING")
    logger.info("="*60)
    
    gb_model = GradientBoostingModel()
    gb_model.train(X_train, y_train)
    gb_model.evaluate(X_test, y_test)
    gb_model.print_metrics()
    
    if save_figure:
        gb_model.plot_results(figure_path)
    else:
        gb_model.plot_results()
    
    if save_model:
        gb_model.save_model(model_path)
    
    return gb_model


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    train_gradient_boosting(save_model=True, save_figure=True)