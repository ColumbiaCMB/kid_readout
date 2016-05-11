import logging
import logging.handlers
default_handler = logging.handlers.StreamHandler()
default_formatter = logging.Formatter('%(levelname)s: %(asctime)s - %(name)s.%(funcName)s:%(lineno)d  %(message)s')
default_handler.setFormatter(default_formatter)
