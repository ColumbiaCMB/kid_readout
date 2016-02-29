__author__ = 'gjones'

"""
None of the tests in this module should require actual hardware
"""
import kid_readout.roach.tests.mock_roach
import kid_readout.roach.heterodyne
import kid_readout.roach.baseband
import numpy as np

def test_calc_fft_bins():
    mr = kid_readout.roach.tests.mock_roach.MockRoach('roach')
    np.random.seed(123)
    for class_ in [kid_readout.roach.heterodyne.RoachHeterodyne,
                   kid_readout.roach.baseband.RoachBaseband]:
        ri = class_(roach=mr,initialize=False)
        for nsamp in 2**np.arange(10,18):
            tone_bins = np.random.random_integers(0,nsamp,size=128)
            ri.calc_fft_bins(tone_bins,nsamp)
"""
from kid_readout.roach.heterodyne import RoachHeterodyne

class TestHeterodyne():
    @classmethod
    def setup_class(cls):
        print "********** making roach"
        cls.ri = RoachHeterodyne()
        cls.ri.initialize(use_config=False)
    def setup(self):
        print "blanking roach"
        self.ri.r.write_int('sync',0)
    def test_1(self):
        print "reading test_1",self.ri.r.read_int('sync')
        self.ri.r.write_int('sync',2)
    def test_2(self):
        print "reading test_2",self.ri.r.read_int('sync')
        self.ri.r.write_int('sync',2)
"""