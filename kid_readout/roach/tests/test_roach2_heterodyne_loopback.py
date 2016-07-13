"""
This module runs tests on the ROACH2 in heterodyne mode using loopback.
"""

import numpy as np
import time
from kid_readout.roach.r2heterodyne import Roach2Heterodyne
from kid_readout.settings import ROACH2_IP, ROACH2_VALON, ROACH2_HOST_IP
from kid_readout.roach.tests.mock_valon import MockValon
from kid_readout.roach.tests.mixin import RoachMixin, Roach2Mixin, HeterodyneSoftwareMixin, HeterodyneHardwareMixin

# This causes nose test discovery to not add tests found in this module. To run these tests, specify
# $ nosetests test_roach2_heterodyne_loopback.py
__test__ = False


class TestRoach2HeterodyneLoopback(RoachMixin, Roach2Mixin, HeterodyneSoftwareMixin, HeterodyneHardwareMixin):

    @classmethod
    def setup_class(cls):
        cls.ri = Roach2Heterodyne(roachip=ROACH2_IP, adc_valon=ROACH2_VALON, host_ip=ROACH2_HOST_IP,
                                  lo_valon=MockValon(), initialize=False)
        cls.ri.initialize(use_config=False)

    @classmethod
    def teardown_class(cls):
        cls.ri.initialize(use_config=False)
        cls.ri.r.stop() # be sure to disconnect to avoid too many connections to roach
