import logging
import os
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%d-%m-%Y-%H-%M-%S')}.log"
log_path = os.path.join(os.getcwd(), "logs", LOG_FILE)
os.makedirs(os.path.dirname(log_path), exist_ok=True)

LOG_FILE_PATH = log_path

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format='[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    filemode='w'
)

