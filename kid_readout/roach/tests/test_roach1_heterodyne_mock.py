"""
This module runs tests on the ROACH1 in heterodyne mode using mock hardware.
"""
from kid_readout.roach.heterodyne import RoachHeterodyne
from kid_readout.roach.tests.mock_roach import MockRoach
from kid_readout.roach.tests.mock_valon import MockValon
from kid_readout.roach.tests.mixin import RoachMixin, Roach1Mixin, HeterodyneSoftwareMixin, MockMixin


class TestRoach1HeterodyneMock(RoachMixin, Roach1Mixin, HeterodyneSoftwareMixin, MockMixin):

    @classmethod
    def setup_class(cls):
        cls.ri = RoachHeterodyne(roach=MockRoach('roach'), adc_valon=MockValon(), lo_valon=MockValon(),
                                 initialize=False)
