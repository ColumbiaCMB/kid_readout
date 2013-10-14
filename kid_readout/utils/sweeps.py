import numpy as np
import time

from data_block import DataBlock, SweepData

def fine_sweep(ri,freqs=np.linspace(10,200,384),offs=np.linspace(0,0.5,8),nsamp=2**15,nchan_per_step=4,reads_per_step=2):
    tones = []
    davgs = []
    datas = []
    chanids = []
    for k,off in enumerate(offs):
        tt,davg,data,chids = coarse_sweep(ri,freqs=freqs+off,nsamp=nsamp,nchan_per_step=nchan_per_step,reads_per_step=reads_per_step)
        tones.append(tt)
        davgs.append(davg)
        datas.extend(data)
        chanids.append(chids)
        
    tones = np.concatenate(tones)
    chanids = np.concatenate(chanids)
    order = tones.argsort()
    davgs = np.concatenate(davgs)
    davgs = davgs[order]
    tones = tones[order]
    return tones,davgs,datas,chanids

def coarse_sweep(ri,freqs=np.linspace(10,200,384),nsamp=2**15,nchan_per_step=4,reads_per_step=2, callback = None, sweep_id = 1):
    actual_freqs = ri.set_tone_freqs(freqs,nsamp=nsamp)
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
                     nsamp = nsamp, nfft = ri.nfft, t0 = time.time(), fs = ri.fs)
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