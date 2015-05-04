import numpy as np
from scipy.ndimage import filters
import scipy.signal
from kid_readout.utils.fftfilt import fftfilt
import random

def lpf256(ts):
    return fftfilt(scipy.signal.firwin(256,1/256.),ts)

def medmadmask(ts,thresh=8,axis=0):
    med = np.median(ts,axis=axis)
    deviations = np.abs(ts-med)
    mad = np.median(deviations,axis=axis)
    mask = deviations > thresh*mad
    return mask

def deglitch_block(ts,thresh=5):
    tsl = np.roll(np.abs(fftfilt(scipy.signal.firwin(16,1/16.),ts)),-8)
    mask = medmadmask(tsl,thresh=thresh)
    mask[:-50] = mask[:-50] | mask[50:]
    mask[50:] = mask[50:] | mask[:-50]
    nmask = mask.sum()
#    print "rejecting",nmask/float(ts.shape[0])
    out = ts.copy()
    try:
        out[mask] = np.array(random.sample(ts[~mask],nmask))
    except ValueError:
        print "more masked values than samples to draw from!"    
    return out

def deglitch_window(data, window_length, thresh=6):
    out = np.empty_like(data)
    step = window_length//2
    nstep = data.shape[0]//step
    for k in xrange(nstep):
        start = k-1
        if start < 0:
            start = 0
#        print start*step,(k+1)*step,
        chunk = data[start*step:(k+1)*step]
        res = deglitch_block(chunk,thresh=thresh)
        out[start*step:((start+1)*step)] = res[:step]
    out[start*step:start*step + len(res)] = res
    return out


#the following takes up way too much memory and time, no good.
def despike_full(data, window_length, rejection_threshold=7, preprocess_function=lpf256):
    preproc = np.abs(lpf256(data))
    medians = filters.median_filter(preproc, window_length)
    abs_deviations = np.abs(preproc - medians)
    mads = filters.median_filter(abs_deviations, window_length)
    spike_flags = abs_deviations > (mads*rejection_threshold)
    nspikes = spike_flags.sum()
    data[spike_flags] = np.array(random.sample(data[~spike_flags],nspikes))
    return data