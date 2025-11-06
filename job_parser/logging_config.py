"""
Logging configuration for the Job Description Parser.

This module sets up structured logging with appropriate formatters and handlers
for debugging, monitoring, and error tracking throughout the parser system.
"""

import logging
import sys
from typing import Optional

from .config import CONFIG


def setup_logging(log_level: Optional[str] = None, 
                 log_format: Optional[str] = None) -> logging.Logger:
    """Set up logging configuration for the job parser.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Custom log format string
        
    Returns:
        Configured logger instance for the job parser
        
    Example:
        >>> logger = setup_logging('INFO')
        >>> logger.info('Parser initialized successfully')
    """
    # Use provided level or fall back to config
    level = log_level or CONFIG.log_level
    
    # Default structured log format
    if log_format is None:
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ],
        force=True  # Override any existing configuration
    )
    
    # Create and configure job parser logger
    logger = logging.getLogger('job_parser')
    logger.setLevel(getattr(logging, level.upper()))
    
    # Prevent duplicate logs from propagating to root logger
    logger.propagate = False
    
    # Add console handler if not already present
    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        formatter = logging.Formatter(log_format)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger


def get_logger(name: str = 'job_parser') -> logging.Logger:
    """Get a logger instance for the specified module.
    
    Args:
        name: Logger name, typically the module name
        
    Returns:
        Logger instance configured with the job parser settings
        
    Example:
        >>> logger = get_logger(__name__)
        >>> logger.debug('Processing job description')
    """
    return logging.getLogger(name)


# Initialize default logger
logger = setup_logging()