"""
This module runs tests on the ROACH2 in baseband mode using loopback.
"""
from kid_readout.roach.r2baseband import Roach2Baseband
from kid_readout.settings import ROACH2_IP, ROACH2_VALON, ROACH2_HOST_IP
from kid_readout.roach.tests.mixin import RoachMixin, Roach2Mixin, BasebandSoftwareMixin, BasebandHardwareMixin

# This causes nose test discovery to not add tests found in this module. To run these tests, specify
# $ nosetests test_roach2_baseband_loopback.py
__test__ = False


class TestRoach2BasebandLoopback(RoachMixin, Roach2Mixin, BasebandSoftwareMixin, BasebandHardwareMixin):

    @classmethod
    def setup_class(cls):
        cls.ri = Roach2Baseband(roachip=ROACH2_IP, adc_valon=ROACH2_VALON, host_ip=ROACH2_HOST_IP, initialize=False)
        cls.ri.initialize(use_config=False)

    @classmethod
    def teardown_class(cls):
        cls.ri.initialize(use_config=False)
        cls.ri.r.stop()