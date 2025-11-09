# logging_config.py
"""
Конфигурация логгирования для RAG CLI

Уровни логгирования:
- DEBUG: Детальная информация для диагностики (все операции)
- INFO: Основные события (запросы, результаты)
- WARNING: Предупреждения (prompt injection, необычное поведение)
- ERROR: Ошибки выполнения
- CRITICAL: Критические ошибки системы
"""

import logging
import os
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler

class LoggingConfig:
    """Класс для управления конфигурацией логгирования"""
    
    # Директории и файлы
    LOG_DIR = "./logs"
    MAIN_LOG = "rag_cli.log"
    ERROR_LOG = "rag_cli_errors.log"
    DEBUG_LOG = "rag_cli_debug.log"
    
    # Уровни логгирования
    CONSOLE_LEVEL = logging.INFO
    FILE_LEVEL = logging.DEBUG
    
    # Форматы
    DETAILED_FORMAT = (
        '%(asctime)s | %(levelname)-8s | %(name)-20s | '
        '%(filename)s:%(lineno)d | %(message)s'
    )
    SIMPLE_FORMAT = '%(asctime)s | %(levelname)-8s | %(message)s'
    
    # Настройки ротации
    MAX_BYTES = 10 * 1024 * 1024  # 10 MB
    BACKUP_COUNT = 5
    
    @classmethod
    def setup_advanced_logging(cls, 
                              console_level=None,
                              file_level=None,
                              enable_debug_file=False):
        """
        Расширенная настройка логгирования с несколькими файлами
        
        Args:
            console_level: Уровень для вывода в консоль
            file_level: Уровень для основного файла
            enable_debug_file: Создавать ли отдельный файл для DEBUG логов
        """
        console_level = console_level or cls.CONSOLE_LEVEL
        file_level = file_level or cls.FILE_LEVEL
        
        # Создаём директорию
        os.makedirs(cls.LOG_DIR, exist_ok=True)
        
        # Основной логгер
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)  # Ловим все логи
        logger.handlers.clear()
        
        # Форматтеры
        detailed_formatter = logging.Formatter(
            fmt=cls.DETAILED_FORMAT,
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        simple_formatter = logging.Formatter(
            fmt=cls.SIMPLE_FORMAT,
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 1. Консольный хендлер
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_level)
        console_handler.setFormatter(simple_formatter)
        logger.addHandler(console_handler)
        
        # 2. Основной файловый хендлер (все уровни >= file_level)
        main_file_handler = RotatingFileHandler(
            os.path.join(cls.LOG_DIR, cls.MAIN_LOG),
            maxBytes=cls.MAX_BYTES,
            backupCount=cls.BACKUP_COUNT,
            encoding='utf-8'
        )
        main_file_handler.setLevel(file_level)
        main_file_handler.setFormatter(detailed_formatter)
        logger.addHandler(main_file_handler)
        
        # 3. Хендлер только для ошибок (ERROR и выше)
        error_file_handler = RotatingFileHandler(
            os.path.join(cls.LOG_DIR, cls.ERROR_LOG),
            maxBytes=cls.MAX_BYTES,
            backupCount=cls.BACKUP_COUNT,
            encoding='utf-8'
        )
        error_file_handler.setLevel(logging.ERROR)
        error_file_handler.setFormatter(detailed_formatter)
        logger.addHandler(error_file_handler)
        
        # 4. Опциональный DEBUG файл
        if enable_debug_file:
            debug_file_handler = RotatingFileHandler(
                os.path.join(cls.LOG_DIR, cls.DEBUG_LOG),
                maxBytes=cls.MAX_BYTES,
                backupCount=cls.BACKUP_COUNT,
                encoding='utf-8'
            )
            debug_file_handler.setLevel(logging.DEBUG)
            debug_file_handler.setFormatter(detailed_formatter)
            logger.addHandler(debug_file_handler)
        
        logger.info("Логгирование настроено успешно")
        logger.info(f"Логи в директории: {cls.LOG_DIR}")
        logger.debug(f"Console level: {logging.getLevelName(console_level)}")
        logger.debug(f"File level: {logging.getLevelName(file_level)}")
        
        return logger
    
    @classmethod
    def setup_daily_rotation(cls):
        """
        Настройка логгирования с ежедневной ротацией
        Файлы именуются по датам: rag_cli_2025-01-15.log
        """
        os.makedirs(cls.LOG_DIR, exist_ok=True)
        
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        logger.handlers.clear()
        
        formatter = logging.Formatter(
            fmt=cls.DETAILED_FORMAT,
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Ротация каждый день в полночь
        daily_handler = TimedRotatingFileHandler(
            os.path.join(cls.LOG_DIR, 'rag_cli.log'),
            when='midnight',
            interval=1,
            backupCount=30,  # Храним 30 дней
            encoding='utf-8'
        )
        daily_handler.setLevel(logging.DEBUG)
        daily_handler.setFormatter(formatter)
        daily_handler.suffix = "%Y-%m-%d"
        logger.addHandler(daily_handler)
        
        # Консольный вывод
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger


# Варианты логгирования
"""
from logging_config import LoggingConfig

# Вариант 1: Базовая настройка с раздельными файлами
logger = LoggingConfig.setup_advanced_logging(
    console_level=logging.INFO,
    file_level=logging.DEBUG,
    enable_debug_file=True  # Создаст отдельный файл для DEBUG
)

# Вариант 2: Ежедневная ротация
logger = LoggingConfig.setup_daily_rotation()

# Вариант 3: Только ошибки в консоль
logger = LoggingConfig.setup_advanced_logging(
    console_level=logging.WARNING,
    file_level=logging.DEBUG
)
"""