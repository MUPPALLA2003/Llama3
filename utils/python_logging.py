import logging
import os
from datetime import datetime

def setup_logger(name:str,log_dir:str='logs',level:int=logging.INFO)-> logging.Logger:

    os.makedirs(log_dir,exist_ok = True)
    log_file = os.path.join(log_dir,f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S_%f')}.log")
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] - %(message)s")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    if not logger.hasHandlers():
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger