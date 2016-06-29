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
    def setup(cls):
        cls.ri = Roach2Heterodyne(roachip=ROACH2_IP, adc_valon=ROACH2_VALON, host_ip=ROACH2_HOST_IP,
                                  lo_valon=MockValon(), initialize=False)
        cls.ri.initialize(use_config=False)

    def test_fft_bin_selection(self):
        test_cases = [#np.array([[16368, 16370, 16372, 16374, 16376, 16379, 16381, 16383, 1,
            #3,     5,     8,    10,    12,    14,    16]]),  #this special case doesn't quite work because of the
            # readout order
                      np.array([[16368, 16370, 16372, 16374, 16376, 16379, 16381, 16383,
                                 8,    10,    12,    14,    16, 18, 20, 22]]),
                      np.array([[7333,7335]]),
                      np.array([[ 9328, 10269, 11210, 12150, 13091, 14032, 14973, 15914,   470,
         1411,  2352,  3293,  4234,  5174,  6115,  7056]]),
                      np.array([[7040, 7042, 7044, 7046, 7048, 7051, 7053, 7055, 7057, 7059, 7061,
        7064, 7066, 7068, 7070, 7072]]),
                      np.array([[8193, 8195, 8197, 8199, 8201, 8203, 8206, 8208, 8210, 8212, 8214,
        8216, 8218, 8220, 8222, 8224, 8160, 8162, 8164, 8166, 8168, 8171,
        8173, 8175, 8177, 8179, 8181, 8183, 8185, 8187, 8189, 8191]])]
        for bin_array in test_cases:
            yield self.check_fft_bin_selection, bin_array

    def check_fft_bin_selection(self,bin_array):
        self.ri.set_debug(True)
        self.ri.fft_bins = bin_array
        self.ri.select_fft_bins(range(self.ri.fft_bins.shape[1]))
        #time.sleep(0.1)
        data,sequence = self.ri.get_data(demod=False)
        assert(np.all(data[0,:].imag.astype('int') == self.ri.fpga_fft_readout_indexes))