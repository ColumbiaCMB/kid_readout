import numpy as np
from corr.katcp_wrapper import FpgaClient
from matplotlib import pyplot as plt
import sys
import time
import valon_synth as vs

if not globals().has_key('r1'):
    r1 = FpgaClient('roach')
    globals()['r1'] = r1
    
# alternative boffile which is not yet working: adcfft14dac14r3_2013_May_31_1457.bof'
def initialize(boffile='adcfft14dac14r2_2013_May_29_1658.bof')
    r1.progdev('') # deprogram the ROACH
    r1.progdev(boffile) #program it
    clk = r1.est_brd_clk()
    print "The ROACH clock is ~ %.1f MHz" % clk
    setFFTGain(0)  # generally use FFTgain = 2**0 = 1
    r1.write_int('sync',0) # give a sync pulse to get things going
    r1.write_int('sync',1)
    setChannel(1024) #start up with a tone coming out

def setFFTGain(gain):
    """
    Set the gain in the FFT
    
    At each stage of the FFT there is the option to downshift (divide by 2) the data, reducing the overall
    voltage gain by a factor of 2. Therefore, the FFT gain can only be of the form 2^k for k nonnegative
    
    gain: the number of stages to divide on. The final gain will be 2^gain
    """
    fftshift = (2**20 - 1) - (2**gain - 1)  #this expression puts downsifts at the earliest stages of the FFT
    r1.write_int('fftshift',fftshift)
    
def selectBin(ibin):
    """
    Set the register which selects the FFT bin we get data from
    
    ibin: 0 to fftlen -1
    """
    d,m = divmod(ibin,4)
    offset = 1
    d = (d-offset) % 2**12
    chansel = d + (m<<16)
    r1.write_int('chansel',chansel)
    
def setChannel(ch,dphi=-0.25,amp=-3,nfft=2**14, ns = 2**16):
    """
    ch: channel number (-ns/2 to ns/2)
    dphi: phase offset between I and Q components in turns (nominally 1/4 = pi/2 radians)
    amp: amplitude relative to full scale in dB
    nfft: size of the fft
    ns: number of samples in the playback memory 
    """
    setTone(ch/(1.0*ns), dphi=dphi, amp=amp, ns=ns)
    absch = np.abs(ch)
    chanPerBin = ns/nfft
    ibin = absch // chanPerBin
    if ch < 0:
        ibin = nfft-ibin
    
    selectBin(int(ibin))
    
def getFFT(nread=10):
    a = r1.read_uint('ppout_addr') & 0x1000
    addr = r1.read_uint('ppout_addr') 
    b = addr & 0x1000
    while a == b:
        addr = r1.read_uint('ppout_addr')
        b = addr & 0x1000
    data = []
    addrs = []
    tic = time.time()
    idle = 0
    for n in range(nread):
        a = b
        if a:
            bram = 'ppout_a'
        else:
            bram = 'ppout_b'
        data.append(r1.read(bram,4*2**12))
        addrs.append(addr)
        
        addr = r1.read_uint('ppout_addr')
        b = addr & 0x1000
        while a == b:
            addr = r1.read_uint('ppout_addr')
            b = addr & 0x1000
            idle += 1
        print ("\r got %d" % n),
        sys.stdout.flush()
    tot = time.time()-tic
    print "\rread %d in %.1f seconds, %.2f samples per second, idle %.2f per read" % (nread, tot, (nread*2**12/tot),idle/(nread*1.0))
    dout = np.concatenate(([np.fromstring(x,dtype='>i2').astype('float').view('complex') for x in data]))
    addrs = np.array(addrs)
    return dout,addrs
def getAdc(ns=2**13):
    r1.write_int('i0_ctrl',0)
    r1.write_int('q0_ctrl',0)
    r1.write_int('i0_ctrl',5)
    r1.write_int('q0_ctrl',5)
    s0 = (np.fromstring(r1.read('i0_bram',ns),dtype='>i2'))/16.0
    s1 = (np.fromstring(r1.read('q0_bram',ns),dtype='>i2'))/16.0
    return s0,s1

def getFFTDebug(ns=2**12):
    r1.write_int('fftout_ctrl',0)
    r1.write_int('fftout_ctrl',5)
    r1.write_int('sync',0)
    r1.write_int('sync',1)
    while r1.read_int('fftout_status') != ns*4:
        pass
    d = np.fromstring(r1.read('fftout_bram',ns*4),dtype='>i2').astype('float').view('complex')
    return d
def getIQ(ns=2**17):
    s0,s1 = getAdc(ns=ns)
    return s0+1j*s1
    
def setTone(f0,dphi=0.25,amp=-3,ns = 2**16):
    a = 10**(amp/20.0)
    if a > 0.9999:
        print "warning: clipping amplitude to 0.9999"
        a = 0.9999
    swr = a*np.cos(2*np.pi*(f0*np.arange(ns)))
    swi = a*np.cos(2*np.pi*(dphi+f0*np.arange(ns)))
    qwr = (swr*2**15).astype('>i2')
    qwi = (swi*2**15).astype('>i2')
    r1.blindwrite('iout',qwr.tostring())
    r1.blindwrite('qout',qwi.tostring())
    r1.write_int('dacctrl',0)
    r1.write_int('dacctrl',1)
    return swr+1j*swi

def sweepFFT(chans):
    d = []
    for ch in chans:
        r1.write_int('chansel',ch)
        dout,addr = getFFT(1)
        d.append(dout)
    return np.array(d)

def checkFs(fsr,spacing=10.0):
    d = []
    bc = []
    v = vs.Synthesizer('/dev/ttyUSB1')
    for f in fsr:
        try:
            r1.progdev('')
            try:
                v.set_frequencies(f,f,chan_spacing=spacing)
            except:
                v.conn.close()
                print "skipping",f
                continue
            r1.progdev('adcfft14dac14r2_2013_May_29_1658.bof')
            bc.append(r1.est_brd_clk())
            setTone(-(256)/2.0**14,amp=0)
            s0,s1 = getAdc(ns=2**13)
            d.append(s0+1j*s1)
        except:
            continue
    return d,bc
