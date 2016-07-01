"""
This module runs tests on the ROACH2 in heterodyne mode using mock hardware.
"""
from kid_readout.roach.r2heterodyne import Roach2Heterodyne
from kid_readout.roach.tests.mock_roach import MockRoach
from kid_readout.roach.tests.mock_valon import MockValon
from kid_readout.roach.tests.mixin import RoachMixin, Roach2Mixin, HeterodyneSoftwareMixin, MockMixin


class TestRoach2HeterodyneMock(RoachMixin, Roach2Mixin, HeterodyneSoftwareMixin, MockMixin):

    @classmethod
    def setup_class(cls):
        cls.ri = Roach2Heterodyne(roach=MockRoach('roach'), adc_valon=MockValon(), lo_valon=MockValon(),
                                  initialize=False)
