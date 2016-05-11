import logging
import logging.handlers
default_handler = logging.handlers.StreamHandler()
default_handler.setFormatter(logging.Formatter('%(levelname)s: %(asctime)s - %(name)s.%(funcName)s:%(lineno)d  %(message)s'))
