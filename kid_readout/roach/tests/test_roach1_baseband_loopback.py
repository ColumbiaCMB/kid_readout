"""
This module runs tests on the ROACH1 in baseband mode using loopback.
"""
from kid_readout.roach.baseband import RoachBaseband
from kid_readout.settings import ROACH1_IP, ROACH1_VALON, ROACH1_HOST_IP
from kid_readout.roach.tests.mixin import RoachMixin, Roach1Mixin, BasebandSoftwareMixin, BasebandHardwareMixin

# This causes nose test discovery to not add tests found in this module. To run these tests, specify
# $ nosetests test_roach1_baseband_loopback.py
__test__ = False


class TestRoach1BasebandLoopback(RoachMixin, Roach1Mixin, BasebandSoftwareMixin, BasebandHardwareMixin):

    @classmethod
    def setup(cls):
        cls.ri = RoachBaseband(roachip=ROACH1_IP, adc_valon=ROACH1_VALON, host_ip=ROACH1_HOST_IP, initialize=False)
        cls.ri.initialize(use_config=False)
