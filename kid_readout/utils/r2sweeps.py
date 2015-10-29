__author__ = 'gjones'

import time
import numpy as np
from kid_readout.utils.data_block import SweepData, DataBlock

def do_sweep(ri,center_freqs,offsets,nsamp,
             nchan_per_step=8,reads_per_step=2,callback = None, sweep_data=None,
             demod=True, loopback=False):
    if nchan_per_step > center_freqs.shape[0]:
        nchan_per_step = center_freqs.shape[0]
    if sweep_data is not None:
        swp = sweep_data
    else:
        swp = SweepData()
    for offset_index,offset in enumerate(offsets):
        ri.set_tone_freqs(center_freqs+offset,nsamp=nsamp)
        nchan = ri.fft_bins.shape[1]
        nstep = int(np.ceil(nchan/float(nchan_per_step)))
        toread = set(range(nchan))
        for k in range(nstep):
            if len(toread) == 0:
                break
            if len(toread) < nchan_per_step:
                print "couldn't read last %d channels because number of channels is not a multiple of nchan_per_step" % len(toread)
                break
            if len(toread) > nchan_per_step:
                start = k
                stop = k + (nchan//nstep)*nstep
                selection = range(start,stop,nchan/nchan_per_step)[:nchan_per_step]
                toread = toread.difference(set(selection))
            else:
                selection = list(toread)
                toread = set()
            selection.sort()
            ri.select_fft_bins(selection)
            ri._sync(loopback=loopback)

            time.sleep(0.2)
            epoch = time.time()
            try:
                dmod,addr = ri.get_data_katcp(reads_per_step,demod=demod)
            except Exception,e:
                print e
                continue
            chids = ri.fpga_fft_readout_indexes+1
            tones = ri.tone_bins[0,ri.readout_selection]
            abort = False
            for m in range(len(chids)):
                #print "m:",m,"selection[m]",selection[m],tones[m]*ri.fs*1.0/nsamp,ri.readout_selection[m]
                sweep_index = ri.readout_selection[m]# np.abs(actual_freqs - ri.fs*tones[m]/nsamp).argmin()
                block = DataBlock(data = dmod[:,m], tone=tones[m], fftbin = chids[m],
                         nsamp = nsamp, nfft = ri.nfft, wavenorm = ri.wavenorm, t0 = epoch, fs = ri.fs,
                         sweep_index=sweep_index, heterodyne=ri.heterodyne, lo=ri.lo_frequency,
                         hardware_delay_estimate=ri.hardware_delay_estimate)
                block.progress = (k+1)/float(nstep)
                swp.add_block(block)
                if callback:
                    abort = callback(block)

            if abort:
                break

    return swp