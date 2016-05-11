import logging

default_handler = logging.StreamHandler()
default_formatter = logging.Formatter('%(levelname)s: %(asctime)s - %(name)s.%(funcName)s:%(lineno)d  %(message)s')
default_handler.setFormatter(default_formatter)
