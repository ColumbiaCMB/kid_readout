"""
This module runs tests on the ROACH1 in heterodyne mode using loopback.
"""
from kid_readout.roach.heterodyne import RoachHeterodyne
from kid_readout.settings import ROACH1_IP, ROACH1_VALON, ROACH1_HOST_IP
from kid_readout.roach.tests.mock_valon import MockValon
from kid_readout.roach.tests.mixin import RoachMixin, Roach1Mixin, HeterodyneSoftwareMixin, HeterodyneHardwareMixin

# This causes nose test discovery to not add tests found in this module. To run these tests, specify
# $ nosetests test_roach1_heterodyne_loopback.py
__test__ = False


class TestRoach1HeterodyneLoopback(RoachMixin, Roach1Mixin, HeterodyneSoftwareMixin, HeterodyneHardwareMixin):

    @classmethod
    def setup(cls):
        cls.ri = RoachHeterodyne(roachip=ROACH1_IP, adc_valon=ROACH1_VALON, host_ip=ROACH1_HOST_IP,
                                 lo_valon=MockValon(), initialize=False)
        cls.ri.initialize(use_config=False)
