import uvicorn
import logging
from src.utils.config import config
from src.utils.logger import logger

def main():
    """
    Main entry point for the AI Proctor application.
    Launches the FastAPI app with uvicorn using integrated logging.
    """
    logger.info(f"Starting {config.APP_NAME} v{config.APP_VERSION}")
    logger.info(f"Server will run on {config.HOST}:{config.PORT}")
    
    # Configure uvicorn to use our logger
    log_config = uvicorn.config.LOGGING_CONFIG
    
    # Disable uvicorn's default loggers and use our custom logger
    uvicorn_logger = logging.getLogger("uvicorn")
    uvicorn_access_logger = logging.getLogger("uvicorn.access")
    uvicorn_error_logger = logging.getLogger("uvicorn.error")
    
    # Clear existing handlers
    uvicorn_logger.handlers.clear()
    uvicorn_access_logger.handlers.clear()
    uvicorn_error_logger.handlers.clear()
    
    # Add our logger's handlers to uvicorn loggers
    for handler in logger.handlers:
        uvicorn_logger.addHandler(handler)
        uvicorn_access_logger.addHandler(handler)
        uvicorn_error_logger.addHandler(handler)
    
    # Set log levels
    uvicorn_logger.setLevel(getattr(logging, config.LOG_LEVEL))
    uvicorn_access_logger.setLevel(getattr(logging, config.LOG_LEVEL))
    uvicorn_error_logger.setLevel(getattr(logging, config.LOG_LEVEL))
    
    # Prevent propagation
    uvicorn_logger.propagate = False
    uvicorn_access_logger.propagate = False
    uvicorn_error_logger.propagate = False
    
    # Run uvicorn with custom log config
    uvicorn.run(
        "src.web.app:app",
        host=config.HOST,
        port=config.PORT,
        reload=config.DEBUG,
        log_config=None,  # Disable default logging config
        log_level=config.LOG_LEVEL.lower()
    )

if __name__ == "__main__":
    main()
