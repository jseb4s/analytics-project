import logging
import joblib
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class BankChurnPreprocessor:
    """
    Pipeline de preprocesamiento para datos de churn bancario.
    """
    
    def __init__(self, 
                 columns_to_drop: Optional[List[str]] = None,
                 categorical_cols: Optional[List[str]] = None,
                 numerical_cols: Optional[List[str]] = None):
        """
        Inicializa el preprocesador.
        
        Args:
            columns_to_drop: Columnas a eliminar
            categorical_cols: Columnas categóricas para encoding
            numerical_cols: Columnas numéricas para estandarización
        """
        self.columns_to_drop = columns_to_drop or [
            'clientnum', 'credit_limit', 'total_trans_amt', 
            'total_amt_chng_q4_q1', 'total_revolving_bal'
        ]
        
        self.categorical_cols = categorical_cols or []
        
        self.numerical_cols = numerical_cols or [
            'customer_age', 'dependent_count', 'months_on_book', 
            'total_relationship_count', 'months_inactive_12_mon', 
            'contacts_count_12_mon', 'avg_open_to_buy', 
            'total_trans_ct', 'total_ct_chng_q4_q1', 'avg_utilization_ratio'
        ]
        
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def fit_transform(self, 
                     df: pd.DataFrame,
                     test_size: float = 0.2,
                     random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Preprocesa los datos y divide en train/test.
        
        Args:
            df: DataFrame original
            test_size: Proporción de datos para test
            random_state: Semilla para reproducibilidad
            
        Returns:
            Tupla (X_train, X_test, y_train, y_test)
        """
        logger.info("=== Iniciando preprocesamiento ===")
        logger.info(f"Shape inicial: {df.shape}")
        
        # eliminar columnas
        df_processed = df.drop(columns=self.columns_to_drop)
        logger.info(f"Columnas eliminadas: {self.columns_to_drop}")
        
        # crear variable objetivo
        df_processed['churn'] = (df_processed['attrition_flag'] == 'Attrited Customer').astype(int)
        df_processed = df_processed.drop('attrition_flag', axis=1)
        
        churn_rate = (df_processed['churn'].sum() / len(df_processed)) * 100
        logger.info(f"Tasa de churn: {churn_rate:.2f}%")
        
        # one-hot encoding
        if self.categorical_cols:
            existing_cats = [col for col in self.categorical_cols if col in df_processed.columns]
            df_processed = pd.get_dummies(df_processed, columns=existing_cats, drop_first=True)
            logger.info(f"One-hot encoding aplicado a: {existing_cats}")
        
        # separar X e y
        y = df_processed['churn']
        X = df_processed.drop(columns=['churn'])
        
        logger.info(f"Features (X): {X.shape}, Target (y): {y.shape}")
        
        # train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
        logger.info(f"Train churn rate: {(y_train.sum()/len(y_train)*100):.2f}%")
        logger.info(f"Test churn rate: {(y_test.sum()/len(y_test)*100):.2f}%")
        
        # estandarizar columnas numéricas
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        # verificar que columnas numericas existen
        existing_numerical = [col for col in self.numerical_cols if col in X_train.columns]
        missing_numerical = [col for col in self.numerical_cols if col not in X_train.columns]
        
        if missing_numerical:
            logger.warning(f"Columnas numéricas no encontradas: {missing_numerical}")
        
        if existing_numerical:
            logger.info(f"Estandarizando columnas: {existing_numerical}")
            X_train_scaled[existing_numerical] = self.scaler.fit_transform(X_train[existing_numerical])
            X_test_scaled[existing_numerical] = self.scaler.transform(X_test[existing_numerical])
        
        # guardar nombres de features
        self.feature_names = X_train_scaled.columns.tolist()
        
        logger.info("=== Preprocesamiento completado ===")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def save_scaler(self, filepath: str) -> None:
        """
        Guarda el scaler entrenado.
        
        Args:
            filepath: Ruta donde guardar el scaler
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, filepath)
        logger.info(f"Scaler guardado en: {filepath}")
    
    def load_scaler(self, filepath: str) -> None:
        """
        Carga un scaler previamente entrenado.
        
        Args:
            filepath: Ruta del scaler guardado
        """
        self.scaler = joblib.load(filepath)
        logger.info(f"Scaler cargado desde: {filepath}")
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica la transformacion a nuevos datos (solo estandarizacion).
        
        Args:
            X: DataFrame con features
            
        Returns:
            DataFrame transformado
        """
        X_scaled = X.copy()
        existing_numerical = [col for col in self.numerical_cols if col in X.columns]
        
        if existing_numerical:
            X_scaled[existing_numerical] = self.scaler.transform(X[existing_numerical])
        
        return X_scaled


def save_processed_data(X_train: pd.DataFrame, 
                       X_test: pd.DataFrame,
                       y_train: pd.Series, 
                       y_test: pd.Series,
                       output_dir: Optional[Path] = None) -> None:
    """
    Guarda los datos procesados en archivos CSV.
    
    Args:
        X_train, X_test, y_train, y_test: Datos procesados
        output_dir: Directorio de salida (default: data/processed/)
    """
    if output_dir is None:
        project_root = Path(__file__).parent.parent.parent
        output_dir = project_root / "data" / "processed"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # guardar datasets
    X_train.to_csv(output_dir / "X_train.csv", index=False)
    X_test.to_csv(output_dir / "X_test.csv", index=False)
    y_train.to_csv(output_dir / "y_train.csv", index=False, header=True)
    y_test.to_csv(output_dir / "y_test.csv", index=False, header=True)
    
    logger.info(f"Datos procesados guardados en: {output_dir}")
    logger.info(f"  - X_train.csv: {X_train.shape}")
    logger.info(f"  - X_test.csv: {X_test.shape}")
    logger.info(f"  - y_train.csv: {y_train.shape}")
    logger.info(f"  - y_test.csv: {y_test.shape}")