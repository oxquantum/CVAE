import logging
import os

logger = logging.getLogger(__name__)


def create_directory(directory_path):
    """
    Create the a directory at directory path if it does not already exist
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        logger.info('creating model folder:{}')
