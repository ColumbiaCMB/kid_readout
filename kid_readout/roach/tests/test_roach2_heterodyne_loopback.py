"""
This module runs tests on the ROACH2 in heterodyne mode using loopback.
"""

import numpy as np
import time
from kid_readout.roach.r2heterodyne import Roach2Heterodyne
from kid_readout.settings import ROACH2_IP, ROACH2_VALON, ROACH2_HOST_IP, ROACH2_GBE_HOST_IP
from kid_readout.roach.tests.mock_valon import MockValon
from kid_readout.roach.tests.mixin import RoachMixin, Roach2Mixin, HeterodyneSoftwareMixin, HeterodyneHardwareMixin
from kid_readout.roach import demodulator
from kid_readout.roach import tools
from kid_readout.roach import r2_udp_catcher

# This causes nose test discovery to not add tests found in this module. To run these tests, specify
# $ nosetests test_roach2_heterodyne_loopback.py
__test__ = False


class TestRoach2HeterodyneLoopback(RoachMixin, Roach2Mixin, HeterodyneSoftwareMixin, HeterodyneHardwareMixin):

    @classmethod
    def setup_class(cls):
        cls.ri = Roach2Heterodyne(roachip=ROACH2_IP, adc_valon=ROACH2_VALON, host_ip=ROACH2_GBE_HOST_IP,
                                  lo_valon=MockValon(), initialize=False)
        cls.ri.initialize(use_config=False)
        cls.ri.set_loopback(True)
        cls.ri.set_fft_gain(0)

    @classmethod
    def teardown_class(cls):
        cls.ri.initialize(use_config=False)
        cls.ri.r.stop() # be sure to disconnect to avoid too many connections to roach

    def test_stream_demodulator(self):
        for nsamp in 2**np.arange(15,19):
            for nchan in 2**np.arange(10):
                yield self.check_stream_demodulator_case, nchan, nsamp
    # This test was failing all 40 cases and was disabled on 2017-10-16 by DF and GJ because the StreamDemodulator code
    # was not in use by any of the Roach classes. If this code is ever integrated, this test should be enabled by
    # deleting the line below.
    test_stream_demodulator.__test__ = False

    def check_stream_demodulator_case(self,nchan,nsamp):
        freqs = np.linspace(12.3123, 252.123,num=nchan)
        self.ri.set_tone_baseband_freqs(freqs,nsamp=nsamp, phases=tools.preset_phases(nchan))
        self.ri.select_fft_bins(range(nchan))
        stream_demod = demodulator.get_stream_demodulator_from_roach_state(self.ri.state,self.ri.active_state_arrays)
        for k in range(5):
            blocks = self.ri.blocks_per_second//4
            if blocks < 2:
                blocks = 2
            gold_data,_ = self.ri.get_data(blocks,demod=True)
            packets = r2_udp_catcher.get_udp_packets(self.ri,blocks)
            packets = packets[:-1]
            seq_num,demod_data = stream_demod.decode_and_demodulate_packets(packets,assume_not_contiguous=True)
            demod_data = demod_data.reshape((-1,nchan))

#            raw_data,raw_seqno = self.ri.get_data(self.ri.blocks_per_second//4,demod=False)
#            demod_data = stream_demod.demodulate_stream(raw_data,raw_seqno-self.ri.phase0)

            assert(np.all(np.abs(gold_data-demod_data) < 2))
            assert(np.all(np.abs(gold_data-gold_data.mean(0)) < 2))

