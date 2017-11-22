import logging

from kid_readout.utils import log


root_logger = logging.getLogger('kid_readout')
roach_logger = logging.getLogger('kid_readout.roach.interface')
root_logger.setLevel(logging.DEBUG)

stream_handler = log.default_handler
stream_handler.setLevel(logging.INFO)
stream_handler.addFilter(log.RoachFilter())

file_handler = log.file_handler('logtest.py')

root_logger.addHandler(stream_handler)
root_logger.addHandler(file_handler)

root_logger.debug("debug")
root_logger.info("info")
root_logger.warning("warning")
root_logger.error("error")
root_logger.critical("critical")

roach_logger.debug("debug")
roach_logger.info("info")
roach_logger.warning("warning")
roach_logger.error("error")
roach_logger.critical("critical")
