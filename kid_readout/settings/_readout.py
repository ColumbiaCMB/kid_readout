"""
This is template local settings file for the computer `readout` running the STAR Cryo cryostat in 1132.
"""
import time as _time

from kid_readout.analysis.resources import experiments as _experiments

CRYOSTAT = 'STARCryo'
COOLDOWN = _experiments.get_experiment_info_at(_time.time(), cryostat=CRYOSTAT)

TEMPERATURE_LOG_DIR = '/data/readout/SRS'

SRS_TEMPERATURE_SERIAL_PORT = '/dev/serial/by-id/usb-FTDI_USB_to_Serial_Cable_FTGQM0GY-if00-port0'