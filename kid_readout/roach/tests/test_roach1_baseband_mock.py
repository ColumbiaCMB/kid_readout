"""
This module runs tests on the ROACH1 in baseband mode using mock hardware.
"""
from kid_readout.roach.baseband import RoachBaseband
from kid_readout.roach.tests.mock_roach import MockRoach
from kid_readout.roach.tests.mock_valon import MockValon
from kid_readout.roach.tests.mixin import RoachMixin, Roach1Mixin, BasebandSoftwareMixin, MockMixin


class TestRoach1BasebandMock(RoachMixin, Roach1Mixin, BasebandSoftwareMixin, MockMixin):

    @classmethod
    def setup_class(cls):
        cls.ri = RoachBaseband(roach=MockRoach('roach'), adc_valon=MockValon(), initialize=False)
