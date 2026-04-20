from loguru import logger
import sys

_is_initialized = False

def setup_logger():
    global _is_initialized
    if _is_initialized:
        return
        
    logger.remove()
    
    format_1="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"    
    format_2="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{message}</level>"    
    # Консоль
    logger.add(sys.stderr, format=format_2, level="INFO")
    
    # Файл (без удаления старых)
    logger.add("logs/{time:YYYY-MM-DD_HH-mm-ss}.log", level="DEBUG")

    _is_initialized = True

    logger.info(f'Python {sys.version}')

# Инициализируем при первом импорте
setup_logger()