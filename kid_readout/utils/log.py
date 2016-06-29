import os
import time
import logging

from kid_readout.settings import LOG_DIR

message_format = '%(levelname)s: %(asctime)s - %(name)s.%(funcName)s:%(lineno)d  %(message)s'
default_handler = logging.StreamHandler()
default_formatter = logging.Formatter(message_format)
default_handler.setFormatter(default_formatter)


def file_handler(name, level=logging.DEBUG):
    """
    Return a FileHandler that will write to a log file in the default location with a sensible name.

    Parameters
    ----------
    name : str
        A name to identify the log file; a good practice would be to use __file__ from the calling module.
    level : int
        The log level to use.

    Returns
    -------
    logging.FileHandler
    """
    fh = logging.FileHandler(os.path.join(LOG_DIR, '.'.join([time.strftime('%Y-%m-%d_%H%M%S'), name, 'log'])))
    fh.setFormatter(default_formatter)
    fh.setLevel(level)
    return fh


class RoachFilter(object):
    """
    The roach subpackage emits a lot of log messages with level INFO. This class exists to filter them out.
    """

    def __init__(self, level=logging.WARNING):
        self.level = level
        self.records = []

    def filter(self, record):
        self.records.append(record)
        return not (record.name.startswith('kid_readout.roach') and record.levelno < self.level)
