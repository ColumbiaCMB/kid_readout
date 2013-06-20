"""
Classes to interface to single pixel KID readout systems
"""

import numpy as np
import time
import sys

class SinglePixelReadout(object):
    """
    Base class for single pixel readout systems.
    These methods define an abstract interface that can be relied on to be consistent between
    the baseband and heterodyne readout systems
    """
    def __init__(self):
        raise NotImplementedError("Abstract class, instantiate a subclass instead of this class")
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
        
        gain: the number of stages to divide on. The final gain will be 2^gain
        """
        fftshift = (2**20 - 1) - (2**gain - 1)  #this expression puts downsifts at the earliest stages of the FFT
        self.r.write_int('fftshift',fftshift)
        
    def initialize(self):
        """
        Reprogram the ROACH and get things running
        """
        print "Deprogramming"
        self.r.progdev('')
        print "Programming", self.boffile
        self.r.progdev(self.boffile)
        print "FPGA clock rate ~", self.r.est_brd_clk()
        self.setFFTGain(0)
        self.setChannel(1024)
        
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
        self.setAttenuator(attendb,le_bit=0x02)

    def set_dac_attenuator(self,attendb):
        self.setAttenuator(attendb,le_bit=0x01)
    
    def _read_data(self,nread,bufname):
        """
        Low level data reading loop, common to both readouts
        """
        regname = '%s_addr' % bufname
        a = self.r.read_uint(regname) & 0x1000
        addr = self.r.read_uint(regname) 
        b = addr & 0x1000
        while a == b:
            addr = self.r.read_uint(regname)
            b = addr & 0x1000
        data = []
        addrs = []
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
        return dout,addrs


class SinglePixelBaseband(SinglePixelReadout):
    def __init__(self,roach=None,wafer=0,roachip='roach'):
        """
        Class to represent the baseband readout system (low-frequency (150 MHz), no mixers)
        
        roach: an FpgaClient instance for communicating with the ROACH. If not specified,
                will try to instantiate one connected to *roachip*
        wafer: 0 or 1. 
                In baseband mode, each of the two DAC and ADC connections can be used independantly to
                readout a single wafer each. This parameter indicates which connection you want to use.
        roachip: (optional). Network address of the ROACH if you don't want to provide an FpgaClient
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
                    exit("Connection timeout to roach")
            
        self.wafer = wafer
        self.dac_ns = 2**16 # number of samples in the dac buffer
        self.raw_adc_ns = 2**12 # number of samples in the raw ADC buffer
        self.nfft = 2**14
#        self.boffile = 'adcdac2xfft14r4_2013_Jun_13_1717.bof'
        self.boffile = 'adcdac2xfft14r5_2013_Jun_18_1542.bof'
        
    def set_channel(self,ch,dphi=None,amp=-3):
        """
        ch: channel number (0 to nfft-1)
        dphi: phase offset between I and Q components in turns (nominally 1/4 = pi/2 radians)
        amp: amplitude relative to full scale in dB
        nfft: size of the fft
        ns: number of samples in the playback memory 
        """
        self.setTone(ch/(1.0*self.dac_ns), dphi=dphi, amp=amp)
        absch = np.abs(ch)
        chan_per_bin = (self.dac_ns/self.nfft)/2 # divide by 2 because it's a real signal
        ibin = absch // chan_per_bin
#        if ch < 0:
#            ibin = nfft-ibin       
        self.selectBin(int(ibin))
        
    def get_data(self,nread=10):
        """
        Get a stream of data from a single FFT bin
        
        nread: number of 4096 sample frames to read
        
        returns  dout,addrs
        dout: complex data stream. Real and imaginary parts are each 16 bit signed 
                integers (but cast to numpy complex)
        addrs: counter values when each frame was read. Can be used to check that 
                frames are contiguous
        """
        bufname = 'ppout%d' % self.wafer
        return self._readData(nread, bufname)
        
    def load_waveform(self,wave):
        if len(wave) != self.dac_ns:
            raise Exception("Waveform should be %d samples long" % self.dac_ns)
        w2 = wave.astype('>i2').tostring()
        if self.wafer == 0:
            self.r.blindwrite('iout',w2)
        else:
            self.r.blindwrite('qout',w2)
            
        self.r.write_int('dacctrl',0)
        self.r.write_int('dacctrl',1)
        
    def set_tone(self,f0,dphi=None,amp=-3):
        if dphi:
            print "warning: got dphi parameter in setTone; ignoring for baseband readout"
        a = 10**(amp/20.0)
        if a > 0.9999:
            print "warning: clipping amplitude to 0.9999"
            a = 0.9999
        swr = (2**15)*a*np.cos(2*np.pi*(f0*np.arange(self.dac_ns)))
        self.loadWaveform(swr)
        
    def select_bin(self,ibin):
        """
        Set the register which selects the FFT bin we get data from
        
        ibin: 0 to fftlen -1
        """
        offset = 2 # bins are shifted by 2
        ibin = np.mod(ibin-offset,self.nfft)
        self.r.write_int('chansel',ibin)
    


class SinglePixelHeterodyne(SinglePixelReadout):
    def __init__(self,roach=None,roachip='roach'):
        """
        Class to represent the heterodyne readout system (high frequency, 1.5 GHz, with IQ mixers)
        
        roach: an FpgaClient instance for communicating with the ROACH. If not specified,
                will try to instantiate one connected to *roachip*
        roachip: (optional). Network address of the ROACH if you don't want to provide an FpgaClient
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
                    exit("Connection timeout to roach")
            
        self.dac_ns = 2**16 # number of samples in the dac buffer
        self.raw_adc_ns = 2**12 # number of samples in the raw ADC buffer
        self.nfft = 2**14
        self.boffile = 'adcfft14dac14r2_2013_May_29_1658.bof'
        
    def set_channel(self,ch,dphi=-0.25,amp=-3):
        """
        ch: channel number (-nfft/2 to nfft/2-1)
        dphi: phase offset between I and Q components in turns (nominally -1/4 = pi/2 radians)
        amp: amplitude relative to full scale in dB
        nfft: size of the fft
        ns: number of samples in the playback memory 
        """
        self.setTone(ch/(1.0*self.dac_ns), dphi=dphi, amp=amp)
        absch = np.abs(ch)
        chan_per_bin = self.dac_ns/self.nfft
        ibin = absch // chan_per_bin
        if ch < 0:
            ibin = nfft-ibin       
        self.selectBin(int(ibin))
        
    def get_data(self,nread=10):
        """
        Get a stream of data from a single FFT bin
        
        nread: number of 4096 sample frames to read
        
        returns  dout,addrs
        dout: complex data stream. Real and imaginary parts are each 16 bit signed 
                integers (but cast to numpy complex)
        addrs: counter values when each frame was read. Can be used to check that 
                frames are contiguous
        """
        bufname = 'ppout'
        return self._readData(nread, bufname)
        
    def load_waveform(self,iwave,qwave):
        if len(iwave) != self.dac_ns or len(qwave) != self.dac_ns:
            raise Exception("Waveforms should be %d samples long" % self.dac_ns)
        iw2 = iwave.astype('>i2').tostring()
        qw2 = qwave.astype('>i2').tostring()
    
        self.r.blindwrite('iout',iw2)
        self.r.blindwrite('qout',qw2)
            
        self.r.write_int('dacctrl',0)
        self.r.write_int('dacctrl',1)
        
    def set_tone(self,f0,dphi=0.25,amp=-3):
        if dphi:
            print "warning: got dphi parameter in setTone; ignoring for baseband readout"
        a = 10**(amp/20.0)
        if a > 0.9999:
            print "warning: clipping amplitude to 0.9999"
            a = 0.9999
        swr = (2**15)*a*np.cos(2*np.pi*(f0*np.arange(self.dac_ns)))
        swi = (2**15)*a*np.cos(2*np.pi*(dphi+f0*np.arange(self.dac_ns)))
        self.loadWaveform(swr,swi)
        
    def select_bin(self,ibin):
        """
        Set the register which selects the FFT bin we get data from
        
        ibin: 0 to fftlen -1
        """
        self.r.write_int('chansel',ibin)
    
