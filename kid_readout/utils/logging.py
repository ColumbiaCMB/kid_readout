import logging
default_handler = logging.StreamHandler()
default_handler.setFormatter(logging.Formatter('%(levelname)s: %(asctime)s - %(name)s.%(funcName)s:%(lineno)d  %(message)s'))
