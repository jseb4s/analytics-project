import logging
import pandas as pd
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def load_bank_churn_data(
    filename: str = "bank_churn.xlsx",
    data_dir: Optional[Path] = None
) -> pd.DataFrame:
    """
    Carga los datos de bank churn desde un archivo Excel.
    
    Args:
        filename: Nombre del archivo (default: "bank_churn.xlsx")
        data_dir: Directorio donde están los datos. Si es None, usa data/raw/
        
    Returns:
        DataFrame con los datos cargados
        
    Raises:
        FileNotFoundError: Si el archivo no existe
        
    Example:
        >>> df = load_bank_churn_data()
        >>> df = load_bank_churn_data("otro_archivo.xlsx")
    """
    if data_dir is None:
        data_dir = Path("data/raw")
    
    file_path = data_dir / filename
    
    if not file_path.exists():
        logger.error(f"Archivo no encontrado: {file_path}")
        raise FileNotFoundError(
            f"No se encontró el archivo: {file_path}\n"
            f"Asegúrate de que el archivo existe en {data_dir}"
        )
    
    short_path = "/".join(file_path.parts[-3:])
    logger.info(f"Cargando datos desde: {short_path}")
    
    try:
        df = pd.read_excel(file_path)
        logger.info(f"Datos cargados exitosamente: {df.shape[0]} filas, {df.shape[1]} columnas")
        return df
    except Exception as e:
        logger.error(f"Error al cargar el archivo {short_path}: {str(e)}")
        raise


def load_processed_data(filename: str) -> pd.DataFrame:
    """
    Carga datos ya procesados desde data/processed/
    
    Args:
        filename: Nombre del archivo procesado
        
    Returns:
        DataFrame con los datos procesados
        
    Raises:
        FileNotFoundError: Si el archivo no existe
        ValueError: Si el formato de archivo no es soportado
    """
    file_path = Path("data/processed") / filename
    
    if not file_path.exists():
        logger.error(f"Archivo procesado no encontrado: {file_path}")
        raise FileNotFoundError(f"No se encontró el archivo: {file_path}")
    
    short_path = "/".join(file_path.parts[-3:])
    logger.info(f"Cargando datos procesados desde: {short_path}")
    
    try:
        # detectar el tipo de archivo y cargar apropiadamente
        if filename.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        elif filename.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        else:
            logger.error(f"Formato de archivo no soportado: {filename}")
            raise ValueError(f"Formato de archivo no soportado: {filename}")
        
        logger.info(f"Datos procesados cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
        return df
        
    except Exception as e:
        logger.error(f"Error al cargar datos procesados: {str(e)}")
        raise


def save_processed_data(df: pd.DataFrame, filename: str) -> None:
    """
    Guarda datos procesados en data/processed/
    
    Args:
        df: DataFrame a guardar
        filename: Nombre del archivo (incluir extensión)
        
    Raises:
        ValueError: Si el formato de archivo no es soportado
    """
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = processed_dir / filename
    
    short_path = "/".join(file_path.parts[-3:])
    logger.info(f"Guardando datos procesados en: {short_path}")
    logger.debug(f"Shape del DataFrame: {df.shape}")
    
    try:
        if filename.endswith('.csv'):
            df.to_csv(file_path, index=False)
        elif filename.endswith(('.xlsx', '.xls')):
            df.to_excel(file_path, index=False)
        elif filename.endswith('.parquet'):
            df.to_parquet(file_path, index=False)
        else:
            logger.error(f"Formato de archivo no soportado: {filename}")
            raise ValueError(f"Formato de archivo no soportado: {filename}")
        
        logger.info(f"Datos guardados exitosamente: {short_path}")
        
    except Exception as e:
        logger.error(f"Error al guardar datos: {str(e)}")
        raise