import numpy as np
from matplotlib import pyplot as plt

import kid_readout.measurement.io.data_block
import kid_readout.analysis.timeseries.fftfilt


def fft_bin_to_freq(bin,n):
    N = (n-1)//2 + 1
    if bin < N:
        freq = bin / (n * 1.0)
    else:
        freq = ((bin - N) - (n//2)) / (n*1.0)
    return freq

def find_peak_freq(data):
    data_fft = np.fft.fft(data)
    peak = abs(data_fft).argmax()
    n = data.shape[0]
    peak_freq = fft_bin_to_freq(peak,n)
    return peak_freq

def demodulate(data,poly_degree=7,debug=False):
    peak_freq = find_peak_freq(data)
    num_points = data.shape[0]
    demod = data * np.exp(-2j*np.pi*peak_freq*np.arange(num_points))
    demod3 =  kid_readout.timedomain.fftfilt.fftfilt(kid_readout.measurement.io.data_block.lpf,demod)
    interval = num_points//10000
    if interval == 0:
        interval = 1
    t = np.arange(num_points)
    poly = np.polyfit(t[4000::interval],np.unwrap(np.angle(demod3[4000::interval])),poly_degree)
    demod2 = demod * np.exp(-1j*np.polyval(poly,t))
    if debug:
        fig = plt.figure()
        ax = fig.add_subplot(211)
        ax.plot(t[::interval],np.unwrap(np.angle(demod[::interval])))
        ax.plot(t[::interval],np.polyval(poly,t[::interval]),'r',lw=2)
        ax = fig.add_subplot(212)
        ax.plot(t[::interval],np.unwrap(np.angle(demod2[::interval])))
#        print demod3.dtype
#        ax.plot(t[::interval],np.unwrap(np.angle(demod3[::interval])))

    return demod2

def fold(data,period,debug=False):
    num_points = data.shape[0]
    fold1 = data[:period*(num_points//period)].reshape((-1,period))
    fold1 = fold1[1:,:] #drop first row since it's often got garbage
    first_rows = np.abs(fold1[:10,:]).mean(0)
    first_peak = abs(first_rows - first_rows.mean()).argmax()
    if first_peak < 2000:
        fold1 = np.roll(fold1,2000,axis=1)
        first_peak += 2000
    if first_peak > 8000:
        fold1 = np.roll(fold1,-8000,axis=1)
        first_peak = first_peak-8000
    last_rows = np.abs(fold1[-10:,:]).mean(0)
    last_peak = abs(last_rows - last_rows.mean()).argmax()
    slope = (last_peak - first_peak)/float(fold1.shape[0]-10)
    print last_peak, first_peak, slope
    fold2 = np.empty_like(fold1)
    for k in range(fold1.shape[0]):
        fold2[k,:] = np.roll(fold1[k,:],-int(np.round(slope*k)))
    fold2 = fold2 * np.exp(-1j*np.angle(fold2[:,:1000]).mean(1)[:,None])
    return fold2

def deproject_rtl(folded,samples_around_peak=200,debug=False):
    if len(folded.shape) > 1:
        folded = folded.mean(0)
    peak_index = np.abs(folded).argmax()
    folded = np.roll(folded,len(folded)/4-peak_index)
    peak_index = len(folded)/4
    if peak_index < len(folded)/2:
        folded = folded - folded[-len(folded)/10:].mean()
    else:
        folded = folded - folded[:len(folded)/10].mean()
    start = peak_index - samples_around_peak
    if start< 0:
        start = 0
    end = peak_index + samples_around_peak
    if end >= len(folded):
        end = len(folded)-1
    p = np.polyfit(folded[start:end].real,folded[start:end].imag,1)
    deprojection = np.exp(-1j*np.arctan(p[0]))
    if debug:
        fig,axs = plt.subplots(1,1)#,figsize=(10,10))
        axs.plot(folded.real,folded.imag,'.')
        axs.plot(folded[start:end].real,folded[start:end].imag,'r.')
        axs.plot([0,deprojection.real*folded.real.max()],[0,-deprojection.imag*folded.imag.max()],'k',lw=2)
    return folded*deprojection, deprojection