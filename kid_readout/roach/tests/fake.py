"""
This module contains prototype fake roach classes.
"""
from __future__ import division
import os
import time
import numpy as np
from kid_readout.roach.baseband import RoachBaseband
from kid_readout.roach.tests import mock_roach, mock_valon


class FakeRoachBaseband(RoachBaseband):



    def get_data_seconds(self, nseconds, demod=True, pow2=True):
        chan_rate = self.fs * 1e6 / (2 * self.nfft)  # samples per second for one tone_index
        samples_per_channel_per_block = 4096
        seconds_per_block = samples_per_channel_per_block / chan_rate
        blocks = int(np.round(nseconds / seconds_per_block))
        if pow2:
            lg2 = np.round(np.log2(blocks))
            if lg2 < 0:
                lg2 = 0
            blocks = 2 ** lg2
        data = (np.random.standard_normal((blocks * samples_per_channel_per_block, self.num_tones)) +
                1j * np.random.standard_normal((blocks * samples_per_channel_per_block, self.num_tones)))
        #time.sleep(seconds_per_block * blocks)
        seqnos = np.arange(data.shape[0])
        return data, seqnos

    def _load_dram_katcp(self, data, tries=2):
        while tries > 0:
            try:
                self._pause_dram()
                print(data.tostring())
                self._unpause_dram()
                return
            except Exception, e:
                print "failure writing to dram, trying again"
            #                print e
            tries = tries - 1
        raise Exception("Writing to dram failed!")

    def _load_dram_ssh(self, data, offset_bytes=0, datafile='boffiles/dram.bin'):
        offset_blocks = offset_bytes / 512  #dd uses blocks of 512 bytes by default
        print("self._update_bof_pid()")
        self._pause_dram()
        print(data.tostring())
        print(os.path.join(self.nfs_root, datafile))
        dram_file = '/proc/%d/hw/ioreg/dram_memory' % self.bof_pid
        datafile = '/' + datafile
        print('ssh root@%s "dd seek=%d if=%s of=%s"' % (self.roachip, offset_blocks, datafile, dram_file))
        self._unpause_dram()


def make_fake_roach_baseband():
    ri = FakeRoachBaseband(roach=mock_roach.MockRoach(None, None), adc_valon=mock_valon.MockValon(), initialize=False,
                           roachip='fake')
    ri.initialize(start_udp=False)
    return ri
