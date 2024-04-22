import logging.config

logging.config.dictConfig(
    {
        "version": 1,
        "disable_existing_loggers": False,
        "handlers": {
            "console": {
                "level": "INFO",
                "class": "logging.StreamHandler",
                "formatter": "detailed",
            },
            "file": {
                "level": "DEBUG",
                "class": "logging.FileHandler",
                "filename": "ollama_client.log",
                "formatter": "detailed",
            },
        },
        "formatters": {
            "detailed": {"format": "%(asctime)s %(levelname)s %(module)s %(message)s"},
        },
        "loggers": {
            "": {  # root logger
                "handlers": ["console", "file"],
                "level": "DEBUG",
            },
        },
    }
)

logger = logging.getLogger(__name__)
