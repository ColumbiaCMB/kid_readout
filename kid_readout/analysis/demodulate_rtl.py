import numpy as np
from matplotlib import pyplot as plt

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
    interval = num_points//1000
    if interval == 0:
        interval = 1
    t = np.arange(num_points)
    poly = np.polyfit(t[::interval],np.unwrap(np.angle(demod[::interval])),poly_degree)
    demod2 = demod * np.exp(-1j*np.polyval(poly,t))
    if debug:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(t[::interval],np.unwrap(np.angle(demod[::interval])))
        ax.plot(t[::interval],np.polyval(poly,t[::interval]),'r',lw=2)
        ax.plot(t[::interval],np.unwrap(np.angle(demod2[::interval])))

    return demod2
