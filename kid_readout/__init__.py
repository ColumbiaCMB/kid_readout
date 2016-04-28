"""
Top level documentation test
"""

import logging
#  Follow recommendation in the Python logging HOWTO: make sure there is  a handler to avoid "No handlers found"
# error messages. Don't add any other handlers; that's up to the user application (i.e. data taking or analysis scripts)
logging.getLogger(__name__).addHandler(logging.NullHandler())


from kid_readout.roach import heterodyne,baseband,r2heterodyne,r2baseband
from kid_readout.roach.heterodyne import RoachHeterodyne
from kid_readout.roach.baseband import RoachBaseband
from kid_readout.roach.r2baseband import RoachBaseband
from kid_readout.roach.r2heterodyne import Roach2Heterodyne

from kid_readout.analysis.resonator.helpers import *
from kid_readout.measurement.io.helpers import *
from kid_readout.measurement.io.readoutnc import ReadoutNetCDF
from kid_readout.measurement.io.nc import NCFile
from kid_readout.analysis.resonator import lmfit_resonator
from kid_readout.measurement.acquire import acquire
