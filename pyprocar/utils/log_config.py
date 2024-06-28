import os
from datetime import datetime
import logging

class InfoFilter(logging.Filter):
    def filter(self, record):
        # Only allow INFO level messages
        return record.levelno == logging.INFO

def setup_logging(log_dir,apply_filter=False, log_level='debug'):
    """
    Set up logging for the MatGraphDB package.

    Args:
        log_dir (str): The directory where the log file will be saved.

    Returns:
        logger (logging.Logger): The logger object for the MatGraphDB package.
    """
    log_levels = {
                'debug': logging.DEBUG, 
                'info': logging.INFO, 
                'warning': logging.WARNING, 
                'error': logging.ERROR, 
                'critical': logging.CRITICAL}
    if log_level not in ['debug','info','warning','error','critical']:
        raise ValueError(f"""Invalid log level: {log_level}. 
                        Must be one of 'debug', 'info', 
                        'warning', 'error', or 'critical'.""")

    # Create a timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"package_{timestamp}.log"
    log_filename="package.log"

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logging.basicConfig(filename=os.path.join(log_dir,log_filename), 
                        level=logging.DEBUG, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s - Line: %(lineno)d')

    logger = logging.getLogger('pyprocar')  # define globally (used in train.py, val.py, detect.py, etc.)
    # if apply_filter:
    #     logger.addFilter(InfoFilter())  # Apply the custom filter to only log INFO messages
    return logger