import pandas as pd
from pathlib import Path
from typing import Optional


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
        
    Example:
        >>> df = load_bank_churn_data()
        >>> df = load_bank_churn_data("otro_archivo.xlsx")
    """
    if data_dir is None:
        # Obtener el directorio raíz del proyecto (2 niveles arriba de src/utils)
        project_root = Path(__file__).parent.parent.parent
        data_dir = project_root / "data" / "raw"
    
    file_path = data_dir / filename
    
    if not file_path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo: {file_path}\n"
            f"Asegúrate de que el archivo existe en {data_dir}"
        )
    short_path = "/".join(file_path.parts[-3:])
    print(f"Cargando datos desde: {short_path}")
    df = pd.read_excel(file_path)
    print(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
    
    return df


def load_processed_data(filename: str) -> pd.DataFrame:
    """
    Carga datos ya procesados desde data/processed/
    
    Args:
        filename: Nombre del archivo procesado
        
    Returns:
        DataFrame con los datos procesados
    """
    project_root = Path(__file__).parent.parent.parent
    file_path = project_root / "data" / "processed" / filename
    
    if not file_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {file_path}")
    
    # Detectar el tipo de archivo y cargar apropiadamente
    if filename.endswith('.csv'):
        return pd.read_csv(file_path)
    elif filename.endswith(('.xlsx', '.xls')):
        return pd.read_excel(file_path)
    elif filename.endswith('.parquet'):
        return pd.read_parquet(file_path)
    else:
        raise ValueError(f"Formato de archivo no soportado: {filename}")


def save_processed_data(df: pd.DataFrame, filename: str) -> None:
    """
    Guarda datos procesados en data/processed/
    
    Args:
        df: DataFrame a guardar
        filename: Nombre del archivo (incluir extensión)
    """
    project_root = Path(__file__).parent.parent.parent
    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = processed_dir / filename
    
    if filename.endswith('.csv'):
        df.to_csv(file_path, index=False)
    elif filename.endswith(('.xlsx', '.xls')):
        df.to_excel(file_path, index=False)
    elif filename.endswith('.parquet'):
        df.to_parquet(file_path, index=False)
    else:
        raise ValueError(f"Formato de archivo no soportado: {filename}")
    
    print(f"Datos guardados en: {file_path}")