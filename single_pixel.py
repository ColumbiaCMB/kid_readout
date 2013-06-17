"""
Classes to interface to single pixel KID readout systems
"""

class SinglePixelReadout(object):
    def getRawAdc(self):
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
    
    def setFFTGain(self,gain):
        """
        Set the gain in the FFT
        
        At each stage of the FFT there is the option to downshift (divide by 2) the data, reducing the overall
        voltage gain by a factor of 2. Therefore, the FFT gain can only be of the form 2^k for k nonnegative
        
        gain: the number of stages to divide on. The final gain will be 2^gain
        """
        fftshift = (2**20 - 1) - (2**gain - 1)  #this expression puts downsifts at the earliest stages of the FFT
        self.r.write_int('fftshift',fftshift)

class SinglePixelBaseband(SinglePixelReadout):
    def __init__(self,roach=None,wafer=0,roachip='roach'):
        if roach:
            self.r = roach
        else:
            from corr.katcp_wrapper import FpgaClient
            self.r = FpgaClient(roachip)
            
        self.wafer = wafer
        self.dac_ns = 2**16 # number of samples in the dac buffer
        self.raw_adc_ns = 2**12 # number of samples in the raw ADC buffer
        self.nfft = 2**14
        
    def setChannel(self,ch,dphi=None,amp=-3):
        """
        ch: channel number (0 to nfft-1)
        dphi: phase offset between I and Q components in turns (nominally 1/4 = pi/2 radians)
        amp: amplitude relative to full scale in dB
        nfft: size of the fft
        ns: number of samples in the playback memory 
        """
        setTone(ch/(1.0*self.dac_ns), dphi=dphi, amp=amp)
        absch = np.abs(ch)
        chan_per_bin = self.dac_ns/self.nfft
        ibin = absch // chan_per_bin
#        if ch < 0:
#            ibin = nfft-ibin       
        self.selectBin(int(ibin))
        
    def getFFT(self,nread=10):
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
        tot = time.time()-tic
        print "\rread %d in %.1f seconds, %.2f samples per second, idle %.2f per read" % (nread, tot, (nread*2**12/tot),idle/(nread*1.0))
        dout = np.concatenate(([np.fromstring(x,dtype='>i2').astype('float').view('complex') for x in data]))
        addrs = np.array(addrs)
        return dout,addrs
        
    def loadWaveform(self,wave):
        if len(wave) != self.dac_ns:
            raise Exception("Waveform should be %d samples long" % self.dac_ns)
        w2 = wave.astype('>i2').tostring()
        if self.wafer == 0:
            self.r.blindwrite('iout',w2)
        else:
            self.r.blindwrite('qout',w2)
            
        self.r.write_int('dacctrl',0)
        self.r.write_int('dacctrl',1)
        
    def setTone(self,f0,dphi=None,amp=-3):
        if dphi:
            print "warning: got dphi parameter in setTone; ignoring for baseband readout"
        a = 10**(amp/20.0)
        if a > 0.9999:
            print "warning: clipping amplitude to 0.9999"
            a = 0.9999
        swr = (2**15)*a*np.cos(2*np.pi*(f0*np.arange(self.dac_ns)))
        self.loadWaveform(swr)
        
    def selectBin(self,ibin):
        """
        Set the register which selects the FFT bin we get data from
        
        ibin: 0 to fftlen -1
        """
        self.r.write_int('chansel',ibin)
    
