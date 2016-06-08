"""
Import the most commonly used features to save typing in interactive environments and scripts.
Basically, doing from kid_readout.interactive import * should get you up and running for most cases.
Feel free to add additional imports as you find them helpful.
"""
import logging
logger = logging.getLogger('kid_readout')
from kid_readout.utils.log import default_handler

if default_handler not in logger.handlers:
    logger.addHandler(default_handler)
    logger.setLevel(logging.INFO)
    logger.info("kid_readout logging setup with default stream handler")

from kid_readout.settings import *
from kid_readout.roach import heterodyne, baseband, r2heterodyne, r2baseband, analog, hardware_tools
from kid_readout.roach.heterodyne import RoachHeterodyne
from kid_readout.roach.baseband import RoachBaseband
from kid_readout.roach.r2baseband import RoachBaseband
from kid_readout.roach.r2heterodyne import Roach2Heterodyne
from kid_readout.roach.attenuator import Attenuator

from kid_readout.analysis.resonator.helpers import *
from kid_readout.measurement.io.readoutnc import ReadoutNetCDF
from kid_readout.measurement.io.nc import NCFile
from kid_readout.measurement.io.easync import EasyNetCDF4
from kid_readout.analysis.resonator import lmfit_resonator, lmfit_models
from kid_readout.measurement import acquire
from kid_readout.measurement.acquire import new_nc_file
from kid_readout import measurement
from kid_readout.measurement import basic, temperature
