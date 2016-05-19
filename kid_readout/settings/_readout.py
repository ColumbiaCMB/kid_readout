"""
This is template local settings file for the computer `readout` running the STAR Cryo cryostat in 1132.
"""
import time

from kid_readout.analysis.resources import experiments as _experiments

CRYOSTAT = 'STARCryo'
COOLDOWN = _experiments.get_experiment_info_at(time.time(), cryostat=CRYOSTAT)

TEMPERATURE_LOG_DIR = '/data/readout/SRS'
