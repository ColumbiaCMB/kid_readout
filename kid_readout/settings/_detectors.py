"""
This is a template local settings file for the computer `detectors` running the HPD cryostat in 1027.
"""

CRYOSTAT = 'HPD'

TEMPERATURE_LOG_DIR = '/data/adc/cooldown_logs'

# The ROACH2 is running in heterodyne mode with the mark2 valon.
from kid_readout.settings._roach import ROACH2_IP, ROACH2_VALON, ROACH2_HOST_IP, MARK2_VALON
