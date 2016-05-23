"""
This is a template local settings file for the computer `detectors` running the HPD cryostat in 1027.
"""
import time as _time

from kid_readout.analysis.resources import experiments as _experiments

CRYOSTAT = 'HPD'
COOLDOWN = _experiments.get_experiment_info_at(_time.time(), cryostat=CRYOSTAT)

TEMPERATURE_LOG_DIR = '/data/adc/cooldown_logs'
