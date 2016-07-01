"""
This module runs tests on the ROACH2 in baseband mode using mock hardware.
"""
from kid_readout.roach.r2baseband import Roach2Baseband
from kid_readout.roach.tests.mock_roach import MockRoach
from kid_readout.roach.tests.mock_valon import MockValon
from kid_readout.roach.tests.mixin import RoachMixin, Roach2Mixin, BasebandSoftwareMixin, MockMixin


class TestRoach2BasebandMock(RoachMixin, Roach2Mixin, BasebandSoftwareMixin, MockMixin):

    @classmethod
    def setup_class(cls):
        cls.ri = Roach2Baseband(roach=MockRoach('roach'), adc_valon=MockValon(), initialize=False)
