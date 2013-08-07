"""
Classes to interface to ROACH hardware for KID readout systems
"""

import numpy as np
import time
import sys
import os
import borph_utils

class RoachInterface(object):
    """
    Base class for readout systems.
    These methods define an abstract interface that can be relied on to be consistent between
    the baseband and heterodyne readout systems
    """
    def __init__(self):
        raise NotImplementedError("Abstract class, instantiate a subclass instead of this class")
    
    def update_bof_pid(self):
        if self.bof_pid:
            return
        try:
            self.bof_pid = borph_utils.get_bof_pid()
        except Exception,e:
            self.bof_pid = None
            raise e 
    def get_raw_adc(self):
        """
        Grab raw ADC samples
        
        returns: s0,s1
        s0 and s1 are the samples from adc 0 and adc 1 respectively
        Each sample is a 12 bit signed integer (cast to a numpy float)
        """
        self.r.write_int('i0_ctrl',0)
        self.r.write_int('q0_ctrl',0)
        self.r.write_int('i0_ctrl',5)
        self.r.write_int('q0_ctrl',5)
        s0 = (np.fromstring(self.r.read('i0_bram',self.raw_adc_ns*2),dtype='>i2'))/16.0
        s1 = (np.fromstring(self.r.read('q0_bram',self.raw_adc_ns*2),dtype='>i2'))/16.0
        return s0,s1
    
    def set_fft_gain(self,gain):
        """
        Set the gain in the FFT
        
        At each stage of the FFT there is the option to downshift (divide by 2) the data, reducing the overall
        voltage gain by a factor of 2. Therefore, the FFT gain can only be of the form 2^k for k nonnegative
        
        gain: the number of stages to not divide on. The final gain will be 2^gain
        """
        fftshift = (2**20 - 1) - (2**gain - 1)  #this expression puts downsifts at the earliest stages of the FFT
        self.fft_gain = gain
        self.r.write_int('fftshift',fftshift)
        
    def initialize(self, fs=512.0):
        """
        Reprogram the ROACH and get things running
        
        fs: float
            Sampling frequency in MHz
        """
        print "Deprogramming"
        self.r.progdev('')
        self._set_fs(fs)
        print "Programming", self.boffile
        self.r.progdev(self.boffile)
        self.set_fft_gain(1)
        self.r.write_int('dacctrl',0)
        self.r.write_int('dacctrl',1)
        estfs = self.measure_fs()
        if np.abs(fs-estfs) > 2.0:
            print "Warning! FPGA clock may not be locked to sampling clock!"
        print "Requested sampling rate %.1f MHz. Estimated sampling rate %.1f MHz" % (fs,estfs)
        
    def measure_fs(self):
        """
        Estimate the sampling rate
        
        This takes about 2 seconds to run
        returns: fs, the approximate sampling rate in MHz
        """
        return 2*self.r.est_brd_clk() 
        
    def select_fft_bins(self,bins):
        raise NotImplementedError("Abstract base class")
        
    def set_channel(self,ch,dphi=None,amp=-3):
        raise NotImplementedError("Abstract base class")
    def get_data(self,nread=10):
        raise NotImplementedError("Abstract base class")
    def set_tone(self,f0,dphi=None,amp=-3):
        raise NotImplementedError("Abstract base class")
    def select_bin(self,ibin):
        raise NotImplementedError("Abstract base class")
    
    def set_attenuator(self,attendb,gpio_reg='gpioa',data_bit=0x08,clk_bit=0x04,le_bit=0x02):
        atten = int(attendb*2)
        self.r.write_int(gpio_reg, 0x00)
        mask = 0x20
        for j in range(6):
            if atten & mask:
                data=data_bit
            else:
                data = 0x00
            mask = mask>>1
            self.r.write_int(gpio_reg, data)
            self.r.write_int(gpio_reg, data | clk_bit)
            self.r.write_int(gpio_reg, data)
        self.r.write_int(gpio_reg, le_bit)
        self.r.write_int(gpio_reg, 0x00)
        
    def set_adc_attenuator(self,attendb):
        self.set_attenuator(attendb,le_bit=0x02)

    def set_dac_attenuator(self,attendb):
        self.set_attenuator(attendb,le_bit=0x01)
    
    def _set_fs(self,fs):
        """
        Set sampling frequency in MHz
        """
        raise NotImplementedError
    def _read_data(self,nread,bufname):
        """
        Low level data reading loop, common to both readouts
        """
        regname = '%s_addr' % bufname
        chanreg = '%s_chan' % bufname
        a = self.r.read_uint(regname) & 0x1000
        addr = self.r.read_uint(regname) 
        b = addr & 0x1000
        while a == b:
            addr = self.r.read_uint(regname)
            b = addr & 0x1000
        data = []
        addrs = []
        chans = []
        tic = time.time()
        idle = 0
        try:
            for n in range(nread):
                a = b
                if a:
                    bram = '%s_a' % bufname
                else:
                    bram = '%s_b' % bufname
                data.append(self.r.read(bram,4*2**12))
                addrs.append(addr)
                chans.append(self.r.read_int(chanreg))
                
                addr = self.r.read_uint(regname)
                b = addr & 0x1000
                while a == b:
                    addr = self.r.read_uint(regname)
                    b = addr & 0x1000
                    idle += 1
                print ("\r got %d" % n),
                sys.stdout.flush()
        except Exception,e:
            print "read only partway because of error:"
            print e
            print "\n"
        tot = time.time()-tic
        print "\rread %d in %.1f seconds, %.2f samples per second, idle %.2f per read" % (nread, tot, (nread*2**12/tot),idle/(nread*1.0))
        dout = np.concatenate(([np.fromstring(x,dtype='>i2').astype('float').view('complex') for x in data]))
        addrs = np.array(addrs)
        chans = np.array(chans)
        return dout,addrs,chans

class RoachHeterodyne(RoachInterface):
    def __init__(self,roach=None,wafer=0,roachip='roach',adc_valon=None):
        """
        Class to represent the heterodyne readout system (high-frequency (1.5 GHz), IQ mixers)
        
        roach: an FpgaClient instance for communicating with the ROACH. 
                If not specified, will try to instantiate one connected to *roachip*
        wafer: 0
                Not used for heterodyne system
        roachip: (optional). Network address of the ROACH if you don't want to provide an FpgaClient
        adc_valon: a Valon class, a string, or None
                Provide access to the Valon class which controls the Valon synthesizer which provides
                the ADC and DAC sampling clock.
                The default None value will use the valon.find_valon function to locate a synthesizer
                and create a Valon class for you.
                You can alternatively pass a string such as '/dev/ttyUSB0' to specify the port for the
                synthesizer, which will then be used for creating a Valon class.
                Finally, for test suites, you can directly pass a Valon class or a class with the same
                interface.
        """
        if roach:
            self.r = roach
        else:
            from corr.katcp_wrapper import FpgaClient
            self.r = FpgaClient(roachip)
            t1 = time.time()
            timeout = 10
            while not self.r.is_connected():
                if (time.time()-t1) > timeout:
                    raise Exception("Connection timeout to roach")
                time.sleep(0.1)
                
        if adc_valon is None:
            import valon
            ports = valon.find_valons()
            if len(ports) == 0:
                raise Exception("No Valon found!")
            self.adc_valon_port = ports[0]
            self.adc_valon = valon.Synthesizer(ports[0]) #use latest port
        elif type(adc_valon) is str:
            import valon
            self.adc_valon_port = adc_valon
            self.adc_valon = valon.Synthesizer(self.adc_valon_port)
        else:
            self.adc_valon = adc_valon
            
        self.bof_pid = None
        self.roachip = roachip
        self.fs = self.adc_valon.get_frequency_a()
        self.wafer = wafer
        self.dac_ns = 2**16 # number of samples in the dac buffer
        self.raw_adc_ns = 2**12 # number of samples in the raw ADC buffer
        self.nfft = 2**14
        self.boffile = 'iq2xpfb14mcr4_2013_Aug_02_1446.bof'
        self.bufname = 'ppout%d' % wafer
    def pause_dram(self):
        self.r.write_int('dram_rst',0)
    def unpause_dram(self):
        self.r.write_int('dram_rst',2)
    def _load_dram(self,data,tries=2):
        while tries > 0:
            try:
                self.pause_dram()
                self.r.write_dram(data.tostring())
                self.unpause_dram()
                return
            except Exception, e:
                print "failure writing to dram, trying again"
#                print e
            tries = tries - 1
        raise Exception("Writing to dram failed!")
        
    def load_waveforms(self,i_wave,q_wave):
        data = np.zeros((2*i_wave.shape[0],),dtype='>i2')
        data[0::4] = i_wave[::2]
        data[1::4] = i_wave[1::2]
        data[2::4] = q_wave[::2]
        data[3::4] = q_wave[1::2]
        self.r.write_int('dram_mask', data.shape[0]/4 - 1)
        self._load_dram(data)
        
    def set_tone_freqs(self,freqs,nsamp,amps=None):
        bins = np.round((freqs/self.fs)*nsamp).astype('int')
        actual_freqs = self.fs*bins/float(nsamp)
        bins[bins<0] = nsamp + bins[bins<0]
        self.set_tone_bins(bins, nsamp,amps=amps)
        self.fft_bins = self.calc_fft_bins(bins, nsamp)
        if self.fft_bins.shape[0] > 4:
            readout_selection = range(4)
        else:
            readout_selection = range(self.fft_bins.shape[0])   
            
        self.select_fft_bins(readout_selection)
        return actual_freqs

    def set_tone_bins(self,bins,nsamp,amps=None):
        spec = np.zeros((nsamp,),dtype='complex')
        self.tone_bins = bins.copy()
        self.tone_nsamp = nsamp
        phases = np.random.random(len(bins))*2*np.pi
        self.phases = phases.copy()
        if amps is None:
            amps = 1.0
        self.amps = amps
        spec[bins] = amps*np.exp(1j*phases)
        wave = np.fft.ifft(spec)
        max = np.abs(wave.real).max()
        i_wave = np.round((wave.real/max)*(2**15-1024)).astype('>i2')
        q_wave = np.round((wave.imag/max)*(2**15-1024)).astype('>i2')
        self.i_wave = i_wave
        self.q_wave = q_wave
        self.load_waveforms(i_wave,q_wave)
        
    def calc_fft_bins(self,tone_bins,nsamp):
        tone_bins_per_fft_bin = nsamp/(self.nfft) 
        fft_bins = np.round(tone_bins/float(tone_bins_per_fft_bin)).astype('int')
        return fft_bins
    
    def fft_bin_to_index(self,bins):
        top_half = bins > self.nfft/2
        idx = bins.copy()
        idx[top_half] = self.nfft - bins[top_half] + self.nfft/2
        return idx
        
    def select_fft_bins(self,readout_selection):
        offset = 4
        idxs = self.fft_bin_to_index(self.fft_bins[readout_selection])
        order = idxs.argsort()
        idxs = idxs[order]
        self.readout_selection = np.array(readout_selection)[order]
        self.fpga_fft_readout_indexes = idxs
        self.readout_fft_bins = self.fft_bins[self.readout_selection]

        binsel = np.zeros((self.fpga_fft_readout_indexes.shape[0]+1,),dtype='>i4')
        #evenodd = np.mod(self.fpga_fft_readout_indexes,2)
        #binsel[:-1] = np.mod(self.fpga_fft_readout_indexes/2-offset,self.nfft/2)
        #binsel[:-1] += evenodd*2**16
        binsel[:-1] = np.mod(self.fpga_fft_readout_indexes-offset,self.nfft)
        binsel[-1] = -1
        self.r.write('chans',binsel.tostring())
        
    def demodulate_data(self,data):
        demod = np.zeros_like(data)
        t = np.arange(data.shape[0])
        for n,ich in enumerate(self.readout_selection):
            phi0 = self.phases[ich]
            k = self.tone_bins[ich]
            m = self.fft_bins[ich]
            if m >= self.nfft/2:
                sign = 1.0
            else:
                sign = -1.0
            nfft = self.nfft
            ns = self.tone_nsamp
            foffs = (k*nfft - m*ns)/float(ns)
            demod[:,n] = np.exp(sign*1j*(2*np.pi*foffs*t + phi0)) * data[:,n]
            if m >= self.nfft/2:
                demod[:,n] = np.conjugate(demod[:,n])
        return demod
                
    def get_data(self,nread=10,demod=True):
        """
        Get a chunk of data
        
        nread: number of 4096 sample frames to read
        
        demod: should the data be demodulated before returning? Default, yes
        
        returns  dout,addrs
        dout: complex data stream. Real and imaginary parts are each 16 bit signed 
                integers (but cast to numpy complex)
        addrs: counter values when each frame was read. Can be used to check that 
                frames are contiguous
        """
        bufname = 'ppout%d' % self.wafer
        chan_offset = 1
        draw,addr,ch =  self._read_data(nread, bufname)
        if not np.all(ch == ch[0]):
            print "all channel registers not the same; this case not yet supported"
            return draw,addr,ch
        if not np.all(np.diff(addr)<8192):
            print "address skip!"
        nch = self.readout_selection.shape[0]
        dout = draw.reshape((-1,nch))
        shift = np.flatnonzero(self.fpga_fft_readout_indexes/2==(ch[0]-chan_offset))[0] - (nch-1)
        print shift
        dout = np.roll(dout,shift,axis=1)
        if demod:
            dout = self.demodulate_data(dout)
        return dout,addr
    
    def set_lo(self,lomhz=1200.0,chan_spacing=2.0):
        """
        Set the local oscillator frequency for the IQ mixers
        
        lomhz: float, frequency in MHz
        """
        self.adc_valon.set_frequency_b(lomhz,chan_spacing=chan_spacing)
        
    def set_dac_attenuator(self,attendb):
        if attendb > 31.5:
            attena = 31.5
            attenb = attendb - attena
        else:
            attena = attendb
            attenb = 0
        self.set_attenuator(attena,le_bit=0x01)
        self.set_attenuator(attenb,le_bit=0x80)
        
    def set_adc_attenuator(self,attendb):
        self.set_attenuator(attendb,le_bit=0x02)
    
    def _set_fs(self,fs,chan_spacing=2.0):
        """
        Set sampling frequency in MHz
        Note, this should generally not be called without also reprogramming the ROACH
        Use initialize() instead        
        """
        self.adc_valon.set_frequency_a(fs,chan_spacing=chan_spacing)
        self.fs = fs



class RoachBaseband(RoachInterface):
    def __init__(self,roach=None,wafer=0,roachip='roach',adc_valon=None):
        """
        Class to represent the baseband readout system (low-frequency (150 MHz), no mixers)
        
        roach: an FpgaClient instance for communicating with the ROACH. 
                If not specified, will try to instantiate one connected to *roachip*
        wafer: 0 or 1. 
                In baseband mode, each of the two DAC and ADC connections can be used independantly to
                readout a single wafer each. This parameter indicates which connection you want to use.
        roachip: (optional). Network address of the ROACH if you don't want to provide an FpgaClient
        adc_valon: a Valon class, a string, or None
                Provide access to the Valon class which controls the Valon synthesizer which provides
                the ADC and DAC sampling clock.
                The default None value will use the valon.find_valon function to locate a synthesizer
                and create a Valon class for you.
                You can alternatively pass a string such as '/dev/ttyUSB0' to specify the port for the
                synthesizer, which will then be used for creating a Valon class.
                Finally, for test suites, you can directly pass a Valon class or a class with the same
                interface.
        """
        if roach:
            self.r = roach
        else:
            from corr.katcp_wrapper import FpgaClient
            self.r = FpgaClient(roachip)
            t1 = time.time()
            timeout = 10
            while not self.r.is_connected():
                if (time.time()-t1) > timeout:
                    raise Exception("Connection timeout to roach")
                time.sleep(0.1)
                
        if adc_valon is None:
            import valon
            ports = valon.find_valons()
            if len(ports) == 0:
                raise Exception("No Valon found!")
            for port in ports:
                try:
                    self.adc_valon_port = port
                    self.adc_valon = valon.Synthesizer(port)
                    f = self.adc_valon.get_frequency_a()
                    break
                except:
                    pass
        elif type(adc_valon) is str:
            import valon
            self.adc_valon_port = adc_valon
            self.adc_valon = valon.Synthesizer(self.adc_valon_port)
        else:
            self.adc_valon = adc_valon
        
        self.bof_pid = None
        self.roachip = roachip
        self.fs = self.adc_valon.get_frequency_a()
        self.wafer = wafer
        self.dac_ns = 2**16 # number of samples in the dac buffer
        self.raw_adc_ns = 2**12 # number of samples in the raw ADC buffer
        self.nfft = 2**14
        self.boffile = 'bb2xpfb14mcr5_2013_Jul_31_1301.bof'
        self.bufname = 'ppout%d' % wafer
    def pause_dram(self):
        self.r.write_int('dram_rst',0)
    def unpause_dram(self):
        self.r.write_int('dram_rst',2)
    def _load_dram(self,data,tries=2):
        while tries > 0:
            try:
                self.pause_dram()
                self.r.write_dram(data.tostring())
                self.unpause_dram()
                return
            except Exception, e:
                print "failure writing to dram, trying again"
#                print e
            tries = tries - 1
        raise Exception("Writing to dram failed!")
    def _load_dram_ssh(self,data,roach_root='/srv/roach_boot/etch',datafile='boffiles/dram.bin'):
        self.update_bof_pid()
        self.pause_dram()
        data.tofile(os.path.join(roach_root,datafile))
        dram_file = '/proc/%d/hw/ioreg/dram_memory' % self.bof_pid
        datafile = '/' + datafile
        result = borph_utils.check_output(('ssh root@%s "dd if=%s of=%s"' % (self.roachip,datafile,dram_file)),shell=True)
        print result
        self.unpause_dram()
        
    def _load_dram_fast_not_working(self,data,roach_root='/roach_mount'):
        self.update_bof_pid()
        self.pause_dram()
        tic = time.time()
        dram_file = 'proc/%d/hw/ioreg/dram_memory' % self.bof_pid
        dram_file = os.path.join(roach_root,dram_file)
        fh = open(dram_file,'wb')
        fh.write(data.tostring())
#        data.tofile(fh)
        fh.close()
        elapsed = time.time()-tic
        print "wrote %.1f MB in %.1f seconds %.1f MB/s" % (data.size/2.**20,elapsed,data.size/elapsed/2.**20)
        self.unpause_dram()

    def load_waveform(self,wave,fast=True):
        data = np.zeros((2*wave.shape[0],),dtype='>i2')
        offset = self.wafer*2
        data[offset::4] = wave[::2]
        data[offset+1::4] = wave[1::2]
        self.r.write_int('dram_mask', data.shape[0]/4 - 1)
        if fast:
            self._load_dram_ssh(data)
        else:
            self._load_dram(data)
        
    def set_tone_freqs(self,freqs,nsamp,amps=None):
        bins = np.round((freqs/self.fs)*nsamp).astype('int')
        actual_freqs = self.fs*bins/float(nsamp)
        self.set_tone_bins(bins, nsamp,amps=amps)
        self.fft_bins = self.calc_fft_bins(bins, nsamp)
        if self.fft_bins.shape[0] > 8:
            readout_selection = range(8)
        else:
            readout_selection = range(self.fft_bins.shape[0])   
            
        self.select_fft_bins(readout_selection)
        return actual_freqs

    def set_tone_bins(self,bins,nsamp,amps=None):
        spec = np.zeros((nsamp/2+1,),dtype='complex')
        self.tone_bins = bins.copy()
        self.tone_nsamp = nsamp
        phases = np.random.random(len(bins))*2*np.pi
        self.phases = phases.copy()
        if amps is None:
            amps = 1.0
        self.amps = amps
        spec[bins] = amps*np.exp(1j*phases)
        wave = np.fft.irfft(spec)
        self.wavenorm = np.abs(wave).max()
        qwave = np.round((wave/self.wavenorm)*(2**15-1024)).astype('>i2')
        self.qwave = qwave
        self.load_waveform(qwave)
        
    def calc_fft_bins(self,tone_bins,nsamp):
        tone_bins_per_fft_bin = nsamp/(2*self.nfft) # factor of 2 because real signal
        fft_bins = np.round(tone_bins/float(tone_bins_per_fft_bin)).astype('int')
        return fft_bins
    
    def fft_bin_to_index(self,bins):
        top_half = bins > self.nfft/2
        idx = bins.copy()
        idx[top_half] = self.nfft - bins[top_half] + self.nfft/2
        return idx
        
    def select_fft_bins(self,readout_selection):
        offset = 2
        idxs = self.fft_bin_to_index(self.fft_bins[readout_selection])
        order = idxs.argsort()
        idxs = idxs[order]
        self.readout_selection = np.array(readout_selection)[order]
        self.fpga_fft_readout_indexes = idxs
        self.readout_fft_bins = self.fft_bins[self.readout_selection]

        binsel = np.zeros((self.fpga_fft_readout_indexes.shape[0]+1,),dtype='>i4')
        binsel[:-1] = np.mod(self.fpga_fft_readout_indexes-offset,self.nfft)
        binsel[-1] = -1
        self.r.write('chans',binsel.tostring())
        
    def demodulate_data(self,data):
        demod = np.zeros_like(data)
        t = np.arange(data.shape[0])
        for n,ich in enumerate(self.readout_selection):
            phi0 = self.phases[ich]
            k = self.tone_bins[ich]
            m = self.fft_bins[ich]
            if m >= self.nfft/2:
                sign = 1.0
            else:
                sign = -1.0
            nfft = self.nfft
            ns = self.tone_nsamp
            foffs = (2*k*nfft - m*ns)/float(ns)
            demod[:,n] = np.exp(sign*1j*(2*np.pi*foffs*t + phi0)) * data[:,n]
            if m >= self.nfft/2:
                demod[:,n] = np.conjugate(demod[:,n])
        return demod
                
    def get_data(self,nread=10,demod=True):
        """
        Get a chunk of data
        
        nread: number of 4096 sample frames to read
        
        demod: should the data be demodulated before returning? Default, yes
        
        returns  dout,addrs
        dout: complex data stream. Real and imaginary parts are each 16 bit signed 
                integers (but cast to numpy complex)
        addrs: counter values when each frame was read. Can be used to check that 
                frames are contiguous
        """
        bufname = 'ppout%d' % self.wafer
        chan_offset = 1
        draw,addr,ch =  self._read_data(nread, bufname)
        if not np.all(ch == ch[0]):
            print "all channel registers not the same; this case not yet supported"
            return draw,addr,ch
        if not np.all(np.diff(addr)<8192):
            print "address skip!"
        nch = self.readout_selection.shape[0]
        dout = draw.reshape((-1,nch))
        shift = np.flatnonzero(self.fpga_fft_readout_indexes==(ch[0]-chan_offset))[0] - (nch-1)
        print shift
        dout = np.roll(dout,shift,axis=1)
        if demod:
            dout = self.demodulate_data(dout)
        return dout,addr
    
    def _set_fs(self,fs,chan_spacing=2.0):
        """
        Set sampling frequency in MHz
        Note, this should generally not be called without also reprogramming the ROACH
        Use initialize() instead        
        """
        self.adc_valon.set_frequency_a(fs,chan_spacing=chan_spacing)
        self.fs = fs


def test_sweep(ri):
    data = []
    tones = []
    ri.r.write_int('sync',0)
    ri.r.write_int('sync',1)
    ri.r.write_int('sync',0)
    for k in range(ri.fft_bins.shape[0]/4):
        ri.select_fft_bins(range(k*4,(k+1)*4))
        time.sleep(0.1)
        d,addr = ri.get_data(2)
        data.append(d)
        tones.append(ri.tone_bins[ri.readout_selection])
    tones = np.concatenate(tones)
    order = tones.argsort()
    davg = np.concatenate([x.mean(0) for x in data])
    davg = davg[order]
    tones = tones[order]
    return tones,davg,data