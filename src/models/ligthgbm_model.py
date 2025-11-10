import logging
from pathlib import Path
from lightgbm import LGBMClassifier

from .base_model import BaseModel, load_processed_data

logger = logging.getLogger(__name__)


class LightGBMModel(BaseModel):
    """Modelo LightGBM."""
    
    def __init__(self, random_state: int = 42, **lgbm_params):
        super().__init__("LightGBM", random_state)
        self.lgbm_params = lgbm_params
    
    def build_model(self) -> LGBMClassifier:
        """Construye el modelo LightGBM."""
        default_params = {
            'n_estimators': 200,
            'learning_rate': 0.1,
            'max_depth': -1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': self.random_state,
            'verbose': -1
        }
        default_params.update(self.lgbm_params)
        
        logger.info(f"Construyendo LightGBM con parametros: {default_params}")
        return LGBMClassifier(**default_params)


def train_lightgbm(save_model: bool = True,
                   save_figure: bool = True) -> LightGBMModel:
    """
    Entrena modelo LightGBM.
    
    Args:
        save_model: Si True, guarda el modelo entrenado
        save_figure: Si True, guarda la figura
        
    Returns:
        Modelo entrenado
    """
    X_train, X_test, y_train, y_test = load_processed_data()
    
    model_path = Path("trained_models/lightgbm/lgbm_model.pkl")
    figure_path = Path("reports/figures/lightgbm/lgbm_results.png")
    
    logger.info("\n" + "="*60)
    logger.info("ENTRENANDO LIGHTGBM")
    logger.info("="*60)
    
    lgbm_model = LightGBMModel()
    lgbm_model.train(X_train, y_train)
    lgbm_model.evaluate(X_test, y_test)
    lgbm_model.print_metrics()
    
    if save_figure:
        lgbm_model.plot_results(figure_path)
    else:
        lgbm_model.plot_results()
    
    if save_model:
        lgbm_model.save_model(model_path)
    
    return lgbm_model


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    train_lightgbm(save_model=True, save_figure=True)