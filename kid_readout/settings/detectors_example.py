"""
This is an example local settings file for the computer `detectors` running the HPD cryostat in 1027.
"""

CRYOSTAT = 'HPD'

# Glenn, please verify.
TEMPERATURE_LOG_DIR = '/data/adc/cooldown_logs'

# The ROACH2 with mark2 valon is running in heterodyne mode.
from kid_readout.roach.columbia import ROACH_HOST_IP, ROACH2_IP, ROACH2_VALON, MARK2_VALON, ROACH2_HOST_IP
ROACH_IS_HETERODYNE = True

