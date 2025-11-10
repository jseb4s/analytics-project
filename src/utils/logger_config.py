import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: Optional[str] = None,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
    mode: str = 'a'
) -> logging.Logger:
    """
    Configura y retorna un logger.
    
    Args:
        name: Nombre del logger. Si es None, usa el logger raíz
        level: Nivel de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path al archivo de log. Si es None, solo imprime en consola
        format_string: Formato personalizado para los logs
        mode: Modo de escritura ('a' = append, 'w' = overwrite)
        
    Returns:
        Logger configurado
        
    Example:
        >>> logger = setup_logger(__name__, level=logging.DEBUG, mode='w')
        >>> logger.info("Proceso iniciado")
    """
    # formato por defecto
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # crear formatter
    formatter = logging.Formatter(format_string, datefmt='%Y-%m-%d %H:%M:%S')
    
    # obtener logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # evitar duplicar handlers si ya existen
    if logger.handlers:
        return logger
    
    # handler para consola
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # handler para archivo (opcional)
    if log_file:
        # crear directorio de logs si no existe
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, mode=mode, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_project_logger(module_name: str) -> logging.Logger:
    """
    Obtiene un logger configurado para el proyecto.
    
    Args:
        module_name: Nombre del módulo (usa __name__)
        
    Returns:
        Logger configurado
        
    Example:
        >>> logger = get_project_logger(__name__)
    """
    log_file = Path("logs/analytics.log")
    
    return setup_logger(
        name=module_name,
        level=logging.INFO,
        log_file=str(log_file)
    )