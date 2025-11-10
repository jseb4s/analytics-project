import logging
import pandas as pd
from typing import List, Tuple

logger = logging.getLogger(__name__)


def drop_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Elimina columnas especificadas del DataFrame.
    
    Args:
        df: DataFrame original
        columns: Lista de nombres de columnas a eliminar
        
    Returns:
        DataFrame sin las columnas especificadas
    """
    logger.info(f"Eliminando columnas: {columns}")
    df_cleaned = df.drop(columns=columns)
    logger.info(f"Shape después de eliminar columnas: {df_cleaned.shape}")
    return df_cleaned


def create_target_variable(df: pd.DataFrame, 
                          target_col: str = 'attrition_flag',
                          churn_value: str = 'Attrited Customer') -> pd.DataFrame:
    """
    Crea la variable objetivo 'churn' (1 = Churn, 0 = No Churn).
    
    Args:
        df: DataFrame original
        target_col: Nombre de la columna con el flag de attrición
        churn_value: Valor que indica churn en la columna original
        
    Returns:
        DataFrame con la columna 'churn' creada y la original eliminada
    """
    logger.info(f"Creando variable objetivo 'churn' desde '{target_col}'")
    
    df_copy = df.copy()
    df_copy['churn'] = (df_copy[target_col] == churn_value).astype(int)
    df_copy = df_copy.drop(target_col, axis=1)
    
    churn_count = df_copy['churn'].sum()
    total_count = len(df_copy)
    churn_rate = (churn_count / total_count) * 100
    
    logger.info(f"Variable 'churn' creada - Tasa de churn: {churn_rate:.2f}% ({churn_count}/{total_count})")
    
    return df_copy


def encode_categorical_variables(df: pd.DataFrame, 
                                 categorical_cols: List[str],
                                 drop_first: bool = True) -> pd.DataFrame:
    """
    Aplica one-hot encoding a las variables categóricas.
    
    Args:
        df: DataFrame original
        categorical_cols: Lista de columnas categóricas a codificar
        drop_first: Si True, elimina la primera categoría para evitar multicolinealidad
        
    Returns:
        DataFrame con las variables categóricas codificadas
    """
    logger.info(f"Aplicando one-hot encoding a: {categorical_cols}")
    
    # Filtrar solo las columnas que existen en el DataFrame
    existing_cols = [col for col in categorical_cols if col in df.columns]
    missing_cols = [col for col in categorical_cols if col not in df.columns]
    
    if missing_cols:
        logger.warning(f"Columnas no encontradas (se omiten): {missing_cols}")
    
    df_encoded = pd.get_dummies(df, columns=existing_cols, drop_first=drop_first)
    
    logger.info(f"Shape después de encoding: {df_encoded.shape}")
    logger.info(f"Nuevas columnas creadas: {df_encoded.shape[1] - df.shape[1]}")
    
    return df_encoded


def prepare_features_and_target(df: pd.DataFrame, 
                                target_col: str = 'churn') -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separa features (X) y target (y).
    
    Args:
        df: DataFrame con todas las columnas
        target_col: Nombre de la columna objetivo
        
    Returns:
        Tupla (X, y) con features y target
    """
    logger.info(f"Separando features y target ('{target_col}')")
    
    if target_col not in df.columns:
        raise ValueError(f"Columna objetivo '{target_col}' no encontrada en el DataFrame")
    
    y = df[target_col]
    X = df.drop(columns=[target_col])
    
    logger.info(f"X shape: {X.shape}, y shape: {y.shape}")
    
    return X, y