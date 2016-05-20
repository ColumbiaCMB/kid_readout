"""
This is a template local settings file for the computer `readout` running the STAR Cryo cryostat in 1132.
"""
import time as _time

from kid_readout.analysis.resources import experiments as _experiments

from kid_readout.settings._roach import *
CRYOSTAT = 'STARCryo'
COOLDOWN = _experiments.get_experiment_info_at(_time.time(), cryostat=CRYOSTAT)
TEMPERATURE_LOG_DIR = '/data/readout/SRS'
