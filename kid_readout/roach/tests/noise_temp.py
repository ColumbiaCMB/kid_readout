import numpy as np
import pylab as pl
from matplotlib import mlab
from kid_readout.roach import hardware_tools, heterodyne
import time


class NoiseMeasurements(object):

    def run(self):
        self.setup_roach()
        self.check_settings()
        self.noise_adc()
        los = [500, 1000, 1500, 2000, 2500, 3000, 3500]
        self.noise_lo_sweep(los)
        return
    
    def setup_roach(self, N=16, nsamp=2**20, NFFT=2**8, lendata=2**6, Fs=512.e6, atten=None, lof=1000, use_r2=True, use_mk2=True):
        self.N = N
        self.nsamp = nsamp
        self.NFFT = NFFT
        self.lendata = lendata
        self.Fs = Fs
        self.use_r2 = use_r2

        if use_r2:
            self.r = hardware_tools.r2_with_mk2()
        else:
            if use_mk2:
                self.r = hardware_tools.r1_with_mk2()
            else:   
                self.r = hardware_tools.r1_with_mk1()
            self.lendata = 1

        if atten != None: 
            self.r.set_dac_attenuator(atten)

        self.r.set_lo(lof, modulator_lo_power=5, demodulator_lo_power=5)
        self.lo = lof
        self.r.iq_delay = 0
        freqs = self.r.set_tone_baseband_freqs(np.linspace(10, 150, N), nsamp=nsamp)
        self.r.select_fft_bins(range(N))
        self.r._sync()
        return

    def noise_lo_sweep(self, los):
        data = {}
        for lof in los:
            self.r.set_lo(lof, modulator_lo_power=5, demodulator_lo_power=5)
            self.lo = lof
            f, da = self.all_chan_noise(m=1)
            data[lof] = da
        data['freqs'] = f
        return data

    def all_chan_noise(self, m=1):
        N = self.N
        bigps = []
        for k in range(N):
            f, psc = self.noise_data(chan=k, m=m, plot=False)
            bigps.append(psc)
        pl.figure()
        for k in range(N):
            ind = f > 0
            pl.semilogx(f[ind], bigps[k][ind], label='ch'+str(k))
        pl.xlabel('Hz')
        pl.ylabel('dB/Hz')
        name = 'all chans psd' +str(self.lo)
        pl.title(name)
        pl.grid()
        pl.show()
        return f, bigps

    def noise_data(self, chan=0, m=1, plot=False):
        NFFT = self.NFFT
        N = self.N
        lendata = self.lendata
        Fs = self.Fs
        pst = np.zeros(NFFT)
        cnt = 0
        while cnt < m:
            data, addr = self.r.get_data_udp(N*lendata, demod=True)
            if self.use_r2 and ((not np.all(np.diff(addr)==2**21/N)) or data.shape[0]!=256*lendata):
                print "bad"
            else:
                ps, f = mlab.psd(data[:,chan], NFFT=NFFT, Fs=Fs/self.r.nfft)
                pst += ps
                cnt += 1
        pst /= cnt
        pst = 10.*np.log10(pst)
        if plot:
            ind = f > 0
            pl.semilogx(f[ind], pst[ind])
            pl.xlabel('Hz')
            pl.ylabel('dB/Hz')
            pl.title('chan data psd')
            pl.grid()
            pl.show()
        return f, pst

    def noise_adc(self, nt=50):
        NFFT = self.NFFT
        pst = np.zeros(NFFT)
        for k in range(nt):
            x, y = self.r.get_raw_adc()
            f, ps = self.take_psd(x, y, db=False)
            pst += ps
        pst /= nt
        pst = 10.*np.log10(pst)
        self.plot_psd(f, pst, "averaged adc psd")
        return 

    def check_settings(self):
        x, y = self.r.get_raw_adc()
        print "ptp ", x.ptp()
        print "std ", x.std()
        if x.ptp() > 4000:
            print "high x ptp"
        f, ps = self.take_psd(x, y, NFFT=2**10)
        self.plot_psd(f, ps, "check settings")
        return

    def plot_psd(self, f, ps, name=''):
        pl.figure()
        pl.plot(f, ps)
        pl.xlabel('Hz')
        pl.ylabel('dB/Hz')
        pl.title(name)
        pl.grid()
        pl.show()
        return 

    def take_psd(self, x, y, db=True, NFFT=None):
        if NFFT=None:
            NFFT=self.NFFT
        ps, f = mlab.psd(x+1j*y, NFFT=NFFT, Fs=self.Fs)
        if db:
            ps = 10.*np.log10(ps) 
        return f, ps


if __name__ == "__main__":
    nmc = NoiseMeasurements()
    nmc.run()




