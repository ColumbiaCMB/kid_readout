import numpy as np
import time

from data_block import DataBlock, SweepData

default_segments_hz = [#np.arange(0,200e3,4e3)-450e3,
                       #np.arange(200e3,300e3,2e3)-450e3,
#                                  np.arange(300e3,400e3,1e3)-450e3,
                                  np.arange(400e3,500e3,0.5e3)-450e3]

default_segments_mhz = [x/1e6 for x in default_segments_hz]

default_offsets_hz = np.concatenate(default_segments_hz)

def segmented_fine_sweep(ri,center_freqs,segments=default_segments_mhz,nchan_per_step=4,reads_per_step=2,sweep_id=2):
    swp = SweepData(sweep_id)
    for k,offsets in enumerate(segments):
        mindelta = np.abs(np.diff(offsets)).min()
        nsamp = np.ceil(np.log2(ri.fs/mindelta))
        print "segment",k,"log2(nsamp)",nsamp
        for offs in offsets:
            coarse_sweep(ri,center_freqs+offs,nsamp=2**nsamp,sweep_data = swp, nchan_per_step = nchan_per_step, reads_per_step=reads_per_step)
    return swp
    

def fine_sweep(ri,center_freqs, sweep_width = 0.1,npoints =128,nsamp=2**20, sweep_data = None):
    offsets = np.linspace(-sweep_width/2.0, sweep_width/2.0, npoints)
    if sweep_data is not None:
        swp = sweep_data
    else:
        swp = SweepData()
        
    for k,offset in enumerate(offsets):
        print "subsweep",k,"of",npoints
        coarse_sweep(ri, center_freqs + offset, nsamp=nsamp, nchan_per_step=4, reads_per_step=2, callback=None,sweep_id=k,sweep_data=swp)
    return swp

def coarse_sweep(ri,freqs=np.linspace(10,200,384),nsamp=2**15,nchan_per_step=4,reads_per_step=2, callback = None, sweep_id = 1, sweep_data = None):
    actual_freqs = ri.set_tone_freqs(freqs,nsamp=nsamp)
    if sweep_data is not None:
        data = sweep_data
    else:
        data = SweepData(sweep_id)
    ri.r.write_int('sync',0)
    ri.r.write_int('sync',1)
    ri.r.write_int('sync',0)
    time.sleep(1)
    nchan = ri.fft_bins.shape[0]
    nstep = np.ceil(nchan/float(nchan_per_step))
    toread = set(range(nchan))
    for k in range(nstep):
        if len(toread) == 0:
            break
        if len(toread) > nchan_per_step:
            start = k
            stop = k + (nchan//nstep)*nstep
            selection = range(start,stop,nchan/nchan_per_step)[:nchan_per_step]
            toread = toread.difference(set(selection))
        else:
            selection = list(toread)
            toread = set()
        ri.select_fft_bins(selection)
        ri.r.write_int('sync',0)
        ri.r.write_int('sync',1)
        ri.r.write_int('sync',0)

        time.sleep(0.2)
        try:
            dmod,addr = ri.get_data(reads_per_step)
        except:
            continue
        dmod = dmod*ri.wavenorm
        chids = ri.fpga_fft_readout_indexes+1
        tones = ri.tone_bins[ri.readout_selection]
        abort = False
        for m in range(len(chids)):
            block = DataBlock(data = dmod[:,m], tone=tones[m], fftbin = chids[m], 
                     nsamp = nsamp, nfft = ri.nfft, t0 = time.time(), fs = ri.fs, sweep_index=selection[m])
            block.progress = (k+1)/float(nstep)
            data.add_block(block)
            if callback:
                abort = callback(block)

        if abort:
            break
        
    return data

def reduce_catcher(din):
    chandata = {}
    for frame in din:
        for pkt in frame:
            if chandata.has_key(pkt['channel_id']):
                chandata[pkt['channel_id']].append(pkt['data'])
            else:
                chandata[pkt['channel_id']] =[pkt['data']]
    chanids = chandata.keys()
    chanids.sort()
    chanids = np.array(chanids)
    dout = None
    for n,chid in enumerate(chanids):
        darr = np.concatenate(chandata[chid])
        print darr.shape
        if dout is None:
            dout = np.empty((len(darr),len(chanids)),dtype='complex64')
            print dout.shape
        dout[:,n] = darr
    return chanids,dout