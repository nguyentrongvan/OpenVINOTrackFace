import logging
from datetime import datetime
import os 

def getLoggerFile(log_fname, fmode, dict_fol=''):
    if (dict_fol != '') and (not os.path.exists(dict_fol)):
        os.mkdir(dict_fol)
    LOG_PATH = os.path.join(dict_fol, log_fname)
    
    logging.basicConfig(#filename=LOG_PATH,
                        format='%(asctime)s - %(message)s',
                        # filemode=fmode,
                        handlers=[
                        logging.FileHandler(LOG_PATH, mode=fmode),
                        logging.StreamHandler()
                            ])   
    # Creating an object
    logger = logging.getLogger()
    # Setting the threshold of logger to DEBUG
    logger.setLevel(logging.INFO)
    return logger