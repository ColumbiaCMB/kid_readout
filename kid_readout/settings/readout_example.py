"""
This is an example local settings file for the computer `readout` running the STAR Cryo cryostat in 1132.
"""

CRYOSTAT = 'STARCryo'

TEMPERATURE_LOG_DIR = '/data/readout/SRS'

# The ROACH1 is connected in baseband mode.
from kid_readout.roach.columbia import ROACH1_HOST_IP, ROACH1_VALON, ROACH1_IP

