"""
Classes to interface to ROACH hardware for KID readout systems
"""
# Long-term interface changes:
# TODO: switch to properties where possible
# TODO: give more explicit access to the tone banks
# TODO: raise RoachError for Roach-specific violations

# TODO: add explicit integer division and use from __future__ import division
import numpy as np
import time
import sys
import os
import socket
import borph_utils
import udp_catcher

from roach_utils import ntone_power_correction

import scipy.signal

try:
    import numexpr

    have_numexpr = True
except ImportError:
    have_numexpr = False
CONFIG_FILE_NAME = '/home/data/roach_config.npz'


def compute_window(npfb=2 ** 15, taps=2, wfunc=scipy.signal.flattop):
    wv = wfunc(npfb * taps)
    sc = np.sinc(np.arange(npfb * taps) / float(npfb) - taps / 2.0)
    coeff = wv * sc
    mag = np.abs(np.fft.fft(coeff, npfb * taps * 2 ** 5)[:2 ** 7])
    mag = mag / mag.max()
    return mag


class RoachError(Exception):
    """
    This class is raised on Roach-specific errors.
    """
    pass


class RoachInterface(object):
    """
    Base class for readout systems.

    These methods define an abstract interface that can be relied on to be consistent between the baseband and
    heterodyne readout systems.
    """

    def __init__(self):
        raise NotImplementedError("Abstract class, instantiate a subclass instead of this class")

    # FPGA Functions
    def _update_bof_pid(self):
        if self.bof_pid:
            return
        try:
            self.bof_pid = borph_utils.get_bof_pid()
        except Exception, e:
            self.bof_pid = None
            raise e

    def get_raw_adc(self):
        """
        Grab raw ADC samples
        
        returns: s0,s1
        s0 and s1 are the samples from adc 0 and adc 1 respectively
        Each sample is a 12 bit signed integer (cast to a numpy float)
        """
        self.r.write_int('i0_ctrl', 0)
        self.r.write_int('q0_ctrl', 0)
        self.r.write_int('i0_ctrl', 5)
        self.r.write_int('q0_ctrl', 5)
        s0 = (np.fromstring(self.r.read('i0_bram', self.raw_adc_ns * 2), dtype='>i2')) / 16.0
        s1 = (np.fromstring(self.r.read('q0_bram', self.raw_adc_ns * 2), dtype='>i2')) / 16.0
        return s0, s1

    def auto_level_adc(self, goal=-2.0, max_tries=3):
        if self.adc_atten < 0:
            self.set_adc_attenuator(31.5)
        n = 0
        while n < max_tries:
            x, y = self.get_raw_adc()
            dbfs = 20 * np.log10(x.ptp() / 4096.0)  # fraction of full scale
            if np.abs(dbfs - goal) < 1.0:
                print "success at: %.1f dB" % self.adc_atten
                break
            newatten = self.adc_atten + (dbfs - goal)
            print "at: %.1f dBFS, current atten: %.1f dB, next trying: %.1f dB" % (dbfs, self.adc_atten, newatten)
            self.set_adc_attenuator(newatten)
            n += 1

    def set_fft_gain(self, gain):
        """
        Set the gain in the FFT
        
        At each stage of the FFT there is the option to downshift (divide by 2) the data, reducing the overall
        voltage gain by a factor of 2. Therefore, the FFT gain can only be of the form 2^k for k nonnegative
        
        gain: the number of stages to not divide on. The final gain will be 2^gain
        """
        fftshift = (2 ** 20 - 1) - (2 ** gain - 1)  # this expression puts downsifts at the earliest stages of the FFT
        self.fft_gain = gain
        self.r.write_int('fftshift', fftshift)
        self.save_state()

    def set_modulation_output(self, rate='low'):
        """
        rate: can be 'high', 'low', 1-8.
            modulation output signal switching state. 'high' and 'low' set the TTL output to a constant level.
            Integers 1-8 set the switching rate to cycle every 2**k spectra
        returns: float, switching rate in Hz.
        """
        rate_register = 'gpiob'
        if str.lower(str(rate)) == 'low':
            self.r.write_int(rate_register, 0)
            self.modulation_rate = 0
            self.modulation_output = 0
            self.save_state()
            return 0.0
        if str.lower(str(rate)) == 'high':
            self.r.write_int(rate_register, 1)
            self.modulation_rate = 0
            self.modulation_output = 1
            self.save_state()
            return 0.0
        if rate >= 1 and rate <= 8:
            self.r.write_int(rate_register, 10 - rate)
            self.modulation_rate = rate
            self.modulation_output = 2
            self.save_state()
            return self.get_modulation_rate_hz()
        else:
            raise ValueError('Invalid value for rate: got %s, expected one of "high", "low", or 1-8' % str(rate))

    def get_modulation_rate_hz(self):
        if self.modulation_rate == 0:
            return 0.0
        else:
            rate_in_hz = (self.fs * 1e6 / (2 * self.nfft)) / (2 ** self.modulation_rate)
            return rate_in_hz

    def save_state(self):
        np.savez(CONFIG_FILE_NAME,
                 boffile=self.boffile,
                 adc_atten=self.adc_atten,
                 dac_atten=self.dac_atten,
                 bof_pid=self.bof_pid,
                 fft_bins=self.fft_bins,
                 fft_gain=self.fft_gain,
                 tone_nsamp=self.tone_nsamp,
                 tone_bins=self.tone_bins,
                 phases=self.phases,
                 modulation_rate=self.modulation_rate,
                 modulation_output=self.modulation_output)
        try:
            os.chmod(CONFIG_FILE_NAME, 0777)
        except:
            pass

    def initialize(self, fs=512.0, start_udp=True, use_config=True):
        """
        Reprogram the ROACH and get things running
        
        fs: float
            Sampling frequency in MHz
        """
        if use_config:
            try:
                state = np.load(CONFIG_FILE_NAME)
                print "Loaded ROACH state from", CONFIG_FILE_NAME
            except IOError:
                print "Could not load previous state"
                state = None
        else:
            state = None
        if state is not None:
            try:
                self._update_bof_pid()
            except Exception:
                self.bof_pid = None
            if self.bof_pid is None or self.bof_pid != state['bof_pid']:
                print "ROACH configuration does not match saved state"
                state = None
        if state is None or state['boffile'] != self.boffile:
            print "Reinitializing system"
            print "Deprogramming"
            self._set_fs(fs)
            self.r.progdev('')
            print "Programming", self.boffile
            self.r.progdev(self.boffile)
            self.bof_pid = None
            self._update_bof_pid()
            self.set_fft_gain(4)
            self.r.write_int('dacctrl', 0)
            self.r.write_int('dacctrl', 1)
            estfs = self.measure_fs()
            if np.abs(fs - estfs) > 2.0:
                print "Warning! FPGA clock may not be locked to sampling clock!"
            print "Requested sampling rate %.1f MHz. Estimated sampling rate %.1f MHz" % (fs, estfs)
            if start_udp:
                print "starting udp server process on PPC"
                borph_utils.start_server(self.bof_pid)
            self.adc_atten = 31.5
            self.dac_atten = np.nan
            self.fft_bins = None
            self.tone_nsamp = None
            self.tone_bins = None
            self.phases = None
            self.modulation_output = 0
            self.modulation_rate = 0
            self.save_state()
        else:
            self.adc_atten = state['adc_atten'][()]
            self.dac_atten = state['dac_atten'][()]
            self.fft_bins = state['fft_bins']
            self.fft_gain = state['fft_gain'][()]
            self.tone_nsamp = state['tone_nsamp'][()]
            self.tone_bins = state['tone_bins']
            self.phases = state['phases']
            self.modulation_output = state['modulation_output'][()]
            self.modulation_rate = state['modulation_rate'][()]

    def measure_fs(self):
        """
        Estimate the sampling rate
        
        This takes about 2 seconds to run
        returns: fs, the approximate sampling rate in MHz
        """
        return 2 * self.r.est_brd_clk()

        # ### Add back in these abstract methods once the interface stablilizes

    #    def select_fft_bins(self,bins):
    #        raise NotImplementedError("Abstract base class")
    #
    #    def set_channel(self,ch,dphi=None,amp=-3):
    #        raise NotImplementedError("Abstract base class")
    #    def get_data(self,nread=10):
    #        raise NotImplementedError("Abstract base class")
    #    def set_tone(self,f0,dphi=None,amp=-3):
    #        raise NotImplementedError("Abstract base class")
    #    def select_bin(self,ibin):
    #        raise NotImplementedError("Abstract base class")

    def _pause_dram(self):
        self.r.write_int('dram_rst', 0)

    def _unpause_dram(self):
        self.r.write_int('dram_rst', 2)

    # TODO: This should raise a RoachError if data is too large to fit in memory.
    def _load_dram(self, data, start_offset=0, fast=True):
        if fast:
            load_dram = self._load_dram_ssh
        else:
            load_dram = self._load_dram_katcp
        nbytes = data.nbytes
        # PPC can only access 64MB at a time, so need to break the data into chunks of this size
        bank_size = (64 * 2 ** 20)
        nbanks, rem = divmod(nbytes, bank_size)
        if rem:
            nbanks += 1
        if nbanks == 0:
            nbanks = 1
        bank_size_units = bank_size / data.itemsize  # calculate how many entries in the data array fit in one 64 MB bank

        start_offset_bytes = start_offset * data.itemsize
        bank_offset = start_offset_bytes // bank_size
        start_offset_bytes = start_offset_bytes - bank_size * bank_offset
        print "bank_offset=", bank_offset, "start_offset=", start_offset, "start_offset_bytes=", start_offset_bytes
        for bank in range(nbanks):
            print "writing DRAM bank", (bank + bank_offset)
            self.r.write_int('dram_controller', bank + bank_offset)
            load_dram(data[bank * bank_size_units:(bank + 1) * bank_size_units], offset_bytes=start_offset_bytes)

    def _load_dram_katcp(self, data, tries=2):
        while tries > 0:
            try:
                self._pause_dram()
                self.r.write_dram(data.tostring())
                self._unpause_dram()
                return
            except Exception, e:
                print "failure writing to dram, trying again"
            #                print e
            tries = tries - 1
        raise Exception("Writing to dram failed!")

    def _load_dram_ssh(self, data, offset_bytes=0, roach_root='/srv/roach_boot/etch', datafile='boffiles/dram.bin'):
        offset_blocks = offset_bytes / 512  #dd uses blocks of 512 bytes by default
        self._update_bof_pid()
        self._pause_dram()
        data.tofile(os.path.join(roach_root, datafile))
        dram_file = '/proc/%d/hw/ioreg/dram_memory' % self.bof_pid
        datafile = '/' + datafile
        result = borph_utils.check_output(
            ('ssh root@%s "dd seek=%d if=%s of=%s"' % (self.roachip, offset_blocks, datafile, dram_file)), shell=True)
        print result
        self._unpause_dram()

    # TODO: call from the functions that require it so we can stop calling it externally.
    def _sync(self):
        self.r.write_int('sync', 0)
        self.r.write_int('sync', 1)
        self.r.write_int('sync', 0)

    ### Other hardware functions (attenuator, valon)
    def set_attenuator(self, attendb, gpio_reg='gpioa', data_bit=0x08, clk_bit=0x04, le_bit=0x02):
        atten = int(attendb * 2)
        try:
            self.r.write_int(gpio_reg, 0x00)
        except RuntimeError:
            print "ROACH not programmed, cannot set attenuators"
            return
        mask = 0x20
        for j in range(6):
            if atten & mask:
                data = data_bit
            else:
                data = 0x00
            mask = mask >> 1
            self.r.write_int(gpio_reg, data)
            self.r.write_int(gpio_reg, data | clk_bit)
            self.r.write_int(gpio_reg, data)
        self.r.write_int(gpio_reg, le_bit)
        self.r.write_int(gpio_reg, 0x00)

    def set_adc_attenuator(self, attendb):
        print "Warning! ADC attenuator is no longer adjustable. Value will be fixed at 31.5 dB"
        self.adc_atten = 31.5

    #        if attendb <0 or attendb > 31.5:
    #            raise ValueError("ADC Attenuator must be between 0 and 31.5 dB. Value given was: %s" % str(attendb))
    #        self.set_attenuator(attendb,le_bit=0x02)
    #        self.adc_atten = int(attendb*2)/2.0

    def set_dac_attenuator(self, attendb):
        if attendb < 0 or attendb > 63:
            raise ValueError("DAC Attenuator must be between 0 and 63 dB. Value given was: %s" % str(attendb))

        if attendb > 31.5:
            attena = 31.5
            attenb = attendb - attena
        else:
            attena = attendb
            attenb = 0
        self.set_attenuator(attena, le_bit=0x01)
        self.set_attenuator(attenb, le_bit=0x02)
        self.dac_atten = int(attendb * 2) / 2.0
        self.save_state()

    def set_dac_atten(self, attendb):
        """ Alias for set_dac_attenuator """
        return self.set_dac_attenuator(attendb)

    def _set_fs(self, fs):
        """
        Set sampling frequency in MHz
        """
        raise NotImplementedError()

    def _window_response(self, fr):
        res = np.interp(np.abs(fr) * 2 ** 7, np.arange(2 ** 7), self._window_mag)
        res = 1 / res
        return res

    ### Tried and true readout function
    def _read_data(self, nread, bufname, verbose=False):
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
                data.append(self.r.read(bram, 4 * 2 ** 12))
                addrs.append(addr)
                chans.append(self.r.read_int(chanreg))

                addr = self.r.read_uint(regname)
                b = addr & 0x1000
                while a == b:
                    addr = self.r.read_uint(regname)
                    b = addr & 0x1000
                    idle += 1
                if verbose:
                    print ("\r got %d" % n),
                sys.stdout.flush()
        except Exception, e:
            print "read only partway because of error:"
            print e
            print "\n"
        tot = time.time() - tic
        print "\rread %d in %.1f seconds, %.2f samples per second, idle %.2f per read" % (
        nread, tot, (nread * 2 ** 12 / tot), idle / (nread * 1.0))
        dout = np.concatenate(([np.fromstring(x, dtype='>i2').astype('float').view('complex') for x in data]))
        addrs = np.array(addrs)
        chans = np.array(chans)
        return dout, addrs, chans

    def _cont_read_data(self, callback, bufname, verbose=False):
        """
        Low level data reading continuous loop, common to both readouts
        calls "callback" each time a chunk of data is ready
        """
        regname = '%s_addr' % bufname
        chanreg = '%s_chan' % bufname
        a = self.r.read_uint(regname) & 0x1000
        addr = self.r.read_uint(regname)
        b = addr & 0x1000
        while a == b:
            addr = self.r.read_uint(regname)
            b = addr & 0x1000
        tic = time.time()
        idle = 0
        n = 0
        try:
            while True:
                try:
                    a = b
                    if a:
                        bram = '%s_a' % bufname
                    else:
                        bram = '%s_b' % bufname
                    data = self.r.read(bram, 4 * 2 ** 12)
                    addrs = addr
                    chans = self.r.read_int(chanreg)
                    res = callback(data, addrs, chans)
                except Exception, e:
                    print "read only partway because of error:"
                    print e
                    print "\n"
                    res = False
                n += 1
                if res:
                    break
                addr = self.r.read_uint(regname)
                b = addr & 0x1000
                while a == b:
                    addr = self.r.read_uint(regname)
                    b = addr & 0x1000
                    idle += 1
                if verbose:
                    print ("\r got %d" % n),
                sys.stdout.flush()
        except KeyboardInterrupt:
            pass
        tot = time.time() - tic
        print "\rread %d in %.1f seconds, %.2f samples per second, idle %.2f per read" % (
        n, tot, (n * 2 ** 12 / tot), idle / (n * 1.0))


class RoachBaseband(RoachInterface):

    def __init__(self, roach=None, wafer=0, roachip='roach', adc_valon=None, host_ip=None, initialize=True):
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
        host_ip: Override IP address to which the ROACH should send it's data. If left as None,
                the host_ip will be set appropriately based on the hostname.
        initialize: Default True, will call self.initialize() which will try to load state from saved config
                Set to False if you don't want this to happen.
        """
        if roach:
            self.r = roach
        else:
            from corr.katcp_wrapper import FpgaClient
            self.r = FpgaClient(roachip)
            t1 = time.time()
            timeout = 10
            while not self.r.is_connected():
                if (time.time() - t1) > timeout:
                    raise Exception("Connection timeout to roach")
                time.sleep(0.1)

        if adc_valon is None:
            import valon
            ports = valon.find_valons()
            if len(ports) == 0:
                self.adc_valon_port = None
                self.adc_valon = None
                print "Warning: No valon found!"
            else:
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

        if host_ip is None:
            hostname = socket.gethostname()
            if hostname == 'detectors':
                host_ip = '192.168.4.2'
            else:
                host_ip = '192.168.1.1'
        self.host_ip = host_ip
        self.adc_atten = 31.5
        self.dac_atten = -1
        self.fft_gain = 0
        self.fft_bins = None
        self.tone_nsamp = None
        self.tone_bins = None
        self.phases = None
        self.modulation_output = 0
        self.modulation_rate = 0
        self.bof_pid = None
        self.roachip = roachip
        # self.boffile = 'bb2xpfb14mcr5_2013_Jul_31_1301.bof'
        #        self.boffile = 'bb2xpfb14mcr7_2013_Oct_31_1332.bof'
        #        self.boffile = 'bb2xpfb14mcr11_2014_Jan_17_1721.bof'
        #self.boffile = 'bb2xpfb14mcr17_2014_Oct_12_1745.bof'
        self.boffile = 'bb2xpfb14mcr17b_2015_Apr_21_1159.bof'

        if initialize:
            self.initialize()
        else:
            print "Not initializing"
        try:
            self.fs = self.adc_valon.get_frequency_a()
        except:
            print "warning couldn't get valon frequency, assuming 512 MHz"
            self.fs = 512.0
        self.wafer = wafer
        self.dac_ns = 2 ** 16  # number of samples in the dac buffer
        self.raw_adc_ns = 2 ** 12  # number of samples in the raw ADC buffer
        self.nfft = 2 ** 14
        self.bufname = 'ppout%d' % wafer
        self._window_mag = compute_window(npfb=2 * self.nfft, taps=2, wfunc=scipy.signal.flattop)
        self.bank = self.get_current_bank()

    # TODO: this should raise a RoachError or return None if no bank is selected or the Roach isn't programmed.
    def get_current_bank(self):
        """
        Determine what tone bank the ROACH is currently set to use
        """
        try:
            bank_reg = self.r.read_int('dram_bank')
            mask_reg = self.r.read_int('dram_mask')
        except RuntimeError:
            return 0  # this catches the case that the ROACH is not yet programmed
        if mask_reg == 0:
            return 0  # if mask is not set, bank is undefined, so call it 0
        self.bank = bank_reg / (mask_reg + 1)

    # TODO: this selects a waveform but doesn't change fft_readout_selection.
    def select_bank(self, bank):
        dram_addr_per_bank = self.tone_nsamp / 2  # number of dram addresses per bank
        mask_reg = dram_addr_per_bank - 1
        bank_reg = dram_addr_per_bank * bank
        self._pause_dram()
        self.r.write_int('dram_bank', bank_reg)
        self.r.write_int('dram_mask', mask_reg)
        self._unpause_dram()
        self.bank = bank

    def load_waveform(self, wave, start_offset=0, fast=True):
        """
        Load waveform
        
        wave : array of 16-bit (dtype='i2') integers with waveform
        
        fast : boolean
            decide what method for loading the dram 
        """
        data = np.zeros((2 * wave.shape[0],), dtype='>i2')
        offset = self.wafer * 2
        data[offset::4] = wave[::2]
        data[offset + 1::4] = wave[1::2]
        start_offset = start_offset * data.shape[0]
        # self.r.write_int('dram_mask', data.shape[0]/4 - 1)
        self._load_dram(data, start_offset=start_offset, fast=fast)

    def set_tone_freqs(self, freqs, nsamp, amps=None, load=True, normfact=None, readout_selection=None):
        """
        Set the stimulus tones to generate
        
        freqs : array of frequencies in MHz
            For baseband system, these must be positive
        nsamp : int, must be power of 2
            number of samples in the playback buffer. Frequency resolution will be fs/nsamp
        amps : optional array of floats, same length as freqs array
            specify the relative amplitude of each tone. Can set to zero to read out a portion
            of the spectrum with no stimulus tone.
        load : bool (debug only). 
            If false, don't actually load the waveform, just calculate it.
                    
        returns:
        actual_freqs : array of the actual frequencies after quantization based on nsamp
        """
        bins = np.round((freqs / self.fs) * nsamp).astype('int')
        actual_freqs = self.fs * bins / float(nsamp)
        self.set_tone_bins(bins, nsamp, amps=amps, load=load, normfact=normfact)
        self.fft_bins = self.calc_fft_bins(bins, nsamp)
        self.select_bank(0)
        if readout_selection is None:
            readout_selection = np.arange(self.fft_bins.shape[1])
        self.select_fft_bins(readout_selection)
        self.save_state()
        return actual_freqs

    def add_tone_freqs(self, freqs, amps=None, overwrite_last=False):
        if freqs.shape[0] != self.tone_bins.shape[1]:
            raise ValueError("freqs array must contain same number of tones as original waveforms")
        # This is a hack that doesn't handle bank selection at all and may have additional problems.
        if overwrite_last:  # Delete the last waveform and readout selection entry.
            self.tone_bins = self.tone_bins[:-1, :]
            self.fft_bins = self.fft_bins[:-1, :]
        nsamp = self.tone_nsamp
        bins = np.round((freqs / self.fs) * nsamp).astype('int')
        actual_freqs = self.fs * bins / float(nsamp)
        self.add_tone_bins(bins, amps=amps)
        self.fft_bins = np.vstack((self.fft_bins, self.calc_fft_bins(bins, nsamp)))
        self.save_state()
        return actual_freqs

    def set_tone_bins(self, bins, nsamp, amps=None, load=True, normfact=None):
        """
        Set the stimulus tones by specific integer bins
        
        bins : array of bins at which tones should be placed
            For Heterodyne system, negative frequencies should be placed in cannonical FFT order
            If 2d, interpret as (nwaves,ntones)
        nsamp : int, must be power of 2
            number of samples in the playback buffer. Frequency resolution will be fs/nsamp
        amps : optional array of floats, same length as bins array
            specify the relative amplitude of each tone. Can set to zero to read out a portion
            of the spectrum with no stimulus tone.
        load : bool (debug only). If false, don't actually load the waveform, just calculate it.
        """

        if bins.ndim == 1:
            bins.shape = (1, bins.shape[0])
        nwaves = bins.shape[0]
        spec = np.zeros((nwaves, nsamp / 2 + 1), dtype='complex')
        self.tone_bins = bins.copy()
        self.tone_nsamp = nsamp
        phases = np.random.random(bins.shape[1]) * 2 * np.pi
        self.phases = phases.copy()
        if amps is None:
            amps = 1.0
        self.amps = amps
        for k in range(nwaves):
            spec[k, bins[k, :]] = amps * np.exp(1j * phases)
        wave = np.fft.irfft(spec, axis=1)
        self.wavenorm = np.abs(wave).max()
        if normfact is not None:
            wn = (2.0 / normfact) * len(bins) / float(nsamp)
            print "ratio of current wavenorm to optimal:", self.wavenorm / wn
            self.wavenorm = wn
        qwave = np.round((wave / self.wavenorm) * (2 ** 15 - 1024)).astype('>i2')
        qwave.shape = (qwave.shape[0] * qwave.shape[1],)
        self.qwave = qwave
        if load:
            self.load_waveform(qwave)
        self.save_state()

    def add_tone_bins(self, bins, amps=None):
        nsamp = self.tone_nsamp
        spec = np.zeros((nsamp / 2 + 1,), dtype='complex')
        self.tone_bins = np.vstack((self.tone_bins, bins))
        phases = self.phases
        if amps is None:
            amps = 1.0
        # self.amps = amps  # TODO: Need to figure out how to deal with this

        spec[bins] = amps * np.exp(1j * phases)
        wave = np.fft.irfft(spec)
        qwave = np.round((wave / self.wavenorm) * (2 ** 15 - 1024)).astype('>i2')
        # self.qwave = qwave  # TODO: Deal with this, if we ever use it
        start_offset = self.tone_bins.shape[0] - 1
        self.load_waveform(qwave, start_offset=start_offset)
        self.save_state()

    def calc_fft_bins(self, tone_bins, nsamp):
        """
        Calculate the FFT bins in which the tones will fall
        
        tone_bins : array of integers
            the tone bins (0 to nsamp - 1) which contain tones
        
        nsamp : length of the playback bufffer
        
        returns : fft_bins, array of integers. 
        """

        tone_bins_per_fft_bin = nsamp / (2 * self.nfft)  # factor of 2 because real signal
        fft_bins = np.round(tone_bins / float(tone_bins_per_fft_bin)).astype('int')
        return fft_bins

    def fft_bin_to_index(self, bins):
        """
        Convert FFT bins to FPGA indexes
        """
        top_half = bins > self.nfft / 2
        idx = bins.copy()
        idx[top_half] = self.nfft - bins[top_half] + self.nfft / 2
        return idx

    def select_fft_bins(self, readout_selection):
        """
        Select which subset of the available FFT bins to read out
        
        Initially we can only read out from a subset of the FFT bins, so this function selects which bins to read out right now
        This also takes care of writing the selection to the FPGA with the appropriate tweaks
        
        The readout selection is stored to self.readout_selection
        The FPGA readout indexes is stored in self.fpga_fft_readout_indexes
        The bins that we are reading out is stored in self.readout_fft_bins
        
        readout_selection : array of ints
            indexes into the self.fft_bins array to specify the bins to read out
        """
        bank = self.bank
        offset = 2
        idxs = self.fft_bin_to_index(self.fft_bins[bank, readout_selection])
        order = idxs.argsort()
        idxs = idxs[order]
        self.readout_selection = np.array(readout_selection)[order]
        self.fpga_fft_readout_indexes = idxs
        self.readout_fft_bins = self.fft_bins[bank, self.readout_selection]

        binsel = np.zeros((self.fpga_fft_readout_indexes.shape[0] + 1,), dtype='>i4')
        binsel[:-1] = np.mod(self.fpga_fft_readout_indexes - offset, self.nfft)
        binsel[-1] = -1
        self.r.write('chans', binsel.tostring())

    def demodulate_data(self, data):
        """
        Demodulate the data from the FFT bin
        
        This function assumes that self.select_fft_bins was called to set up the necessary class attributes
        
        data : array of complex data
        
        returns : demodulated data in an array of the same shape and dtype as *data*
        """
        bank = self.bank
        demod = np.zeros_like(data)
        t = np.arange(data.shape[0])
        for n, ich in enumerate(self.readout_selection):
            phi0 = self.phases[ich]
            k = self.tone_bins[bank, ich]
            m = self.fft_bins[bank, ich]
            if m >= self.nfft / 2:
                sign = 1.0
            else:
                sign = -1.0
            nfft = self.nfft
            ns = self.tone_nsamp
            foffs = (2 * k * nfft - m * ns) / float(ns)
            wc = self._window_response(foffs / 2.0) * (self.tone_nsamp / 2.0 ** 18)
            demod[:, n] = wc * np.exp(sign * 1j * (2 * np.pi * foffs * t + phi0)) * data[:, n]
            if m >= self.nfft / 2:
                demod[:, n] = np.conjugate(demod[:, n])
        return demod

    def get_data_udp(self, nread=2, demod=True):
        chan_offset = 1
        nch = self.fpga_fft_readout_indexes.shape[0]
        data, seqnos = udp_catcher.get_udp_data(self, npkts=nread * 16 * nch, streamid=1,
                                                chans=self.fpga_fft_readout_indexes + chan_offset,
                                                nfft=self.nfft, addr=(self.host_ip, 12345))  # , stream_reg, addr)
        if demod:
            data = self.demodulate_data(data)
        return data, seqnos

    def get_data(self, nread=2, demod=True):
        return self.get_data_udp(nread=nread, demod=demod)

    def get_data_seconds(self, nseconds, demod=True, pow2=True):
        """
        Capture data for specified length of time (using the udp interface)

        nseconds: Number of seconds

        demod: bool, Should the data be demodulated (default True)

        pow2: bool, If true, force the data length to the nearest power of 2
        """
        chan_rate = self.fs * 1e6 / (2 * self.nfft)  # samples per second for one channel
        samples_per_channel_per_block = 4096
        seconds_per_block = samples_per_channel_per_block / chan_rate
        blocks = int(np.round(nseconds / seconds_per_block))
        if pow2:
            lg2 = np.round(np.log2(blocks))
            if lg2 < 0:
                lg2 = 0
            blocks = 2 ** lg2
        return self.get_data_udp(blocks, demod=demod)

    def get_data_seconds_katcp(self, nseconds, demod=True, pow2=True):
        """
        Capture data for specified length of time using the katcp interface
        
        nseconds: Number of seconds
        
        demod: bool, Should the data be demodulated (default True)
        
        pow2: bool, If true, force the data length to the nearest power of 2
        """
        chan_rate = self.fs * 1e6 / (2 * self.nfft)  # samples per second per channel
        nch = self.fpga_fft_readout_indexes.shape[0]
        seconds_per_block = (1024 * nch) / chan_rate
        blocks = int(np.round(nseconds / seconds_per_block))
        if pow2:
            lg2 = np.round(np.log2(blocks))
            if lg2 < 0:
                lg2 = 0
            blocks = 2 ** lg2
        return self.get_data_katcp(blocks, demod=demod)

    def get_data_katcp(self, nread=10, demod=True):
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
        draw, addr, ch = self._read_data(nread, bufname)
        if not np.all(ch == ch[0]):
            print "all channel registers not the same; this case not yet supported"
            return draw, addr, ch
        if not np.all(np.diff(addr) < 8192):
            print "address skip!"
        nch = self.readout_selection.shape[0]
        dout = draw.reshape((-1, nch))
        shift = np.flatnonzero(self.fpga_fft_readout_indexes == (ch[0] - chan_offset))[0] - (nch - 1)
        dout = np.roll(dout, shift, axis=1)
        if demod:
            dout = self.demodulate_data(dout)
        return dout, addr

    def _set_fs(self, fs, chan_spacing=2.0):
        """
        Set sampling frequency in MHz
        Note, this should generally not be called without also reprogramming the ROACH
        Use initialize() instead        
        """
        if self.adc_valon is None:
            print "Could not set Valon; none available"
            return
        self.adc_valon.set_frequency_a(fs,
                                       chan_spacing=chan_spacing)  # for now the baseband readout uses both valon outputs,
        self.adc_valon.set_frequency_b(fs, chan_spacing=chan_spacing)  # one for ADC, one for DAC
        self.fs = fs


class RoachBasebandWide(RoachBaseband):

    def __init__(self, roach=None, wafer=0, roachip='roach', adc_valon=None, host_ip=None):
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
                if (time.time() - t1) > timeout:
                    raise Exception("Connection timeout to roach")
                time.sleep(0.1)

        if adc_valon is None:
            import valon
            ports = valon.find_valons()
            if len(ports) == 0:
                self.adc_valon_port = None
                self.adc_valon = None
                print "Warning: No valon found!"
            else:
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

        if host_ip is None:
            hostname = socket.gethostname()
            if hostname == 'detectors':
                host_ip = '192.168.4.2'
            else:
                host_ip = '192.168.1.1'

        self.host_ip = host_ip
        self.adc_atten = 31.5
        self.dac_atten = -1
        self.fft_gain = 0
        self.fft_bins = None
        self.tone_nsamp = None
        self.tone_bins = None
        self.phases = None
        self.bof_pid = None
        self.roachip = roachip
        try:
            self.fs = self.adc_valon.get_frequency_a()
        except:
            print "warning couldn't get valon frequency, assuming 512 MHz"
            self.fs = 512.0
        self.wafer = wafer
        self.dac_ns = 2 ** 16  # number of samples in the dac buffer
        self.raw_adc_ns = 2 ** 12  # number of samples in the raw ADC buffer
        self.nfft = 2 ** 11
        # self.boffile = 'bb2xpfb12mcr5_2013_Oct_29_1658.bof'
        #self.boffile = 'bb2xpfb11mcr7_2013_Nov_04_1309.bof'
        #self.boffile = 'bb2xpfb11mcr8_2013_Nov_04_2151.bof'
        self.boffile = 'bb2xpfb11mcr11_2014_Feb_01_1106.bof'
        #self.boffile = 'bb2xpfb11mcr12_2014_Feb_26_1028.bof'
        self.bufname = 'ppout%d' % wafer
        self._window_mag = compute_window(npfb=2 * self.nfft, taps=2, wfunc=scipy.signal.flattop)

    def demodulate_data(self, data):
        """
        Demodulate the data from the FFT bin
        
        This function assumes that self.select_fft_bins was called to set up the necessary class attributes
        
        data : array of complex data
        
        returns : demodulated data in an array of the same shape and dtype as *data*
        """
        bank = self.bank
        demod = np.zeros_like(data)
        t = np.arange(data.shape[0])
        for n, ich in enumerate(self.readout_selection):
            phi0 = self.phases[ich]
            k = self.tone_bins[bank, ich]
            m = self.fft_bins[bank, ich]
            if m >= self.nfft / 2:
                sign = 1.0
            else:
                sign = -1.0
            nfft = self.nfft
            ns = self.tone_nsamp
            foffs = (2 * k * nfft - m * ns) / float(ns)
            wc = self._window_response(foffs / 2) * (self.tone_nsamp / 2.0 ** 18)
            if have_numexpr:
                pi = np.pi
                this_data = data[:, n]
                demod[:, n] = numexpr.evaluate('wc*exp(sign*1j*(2*pi*foffs*t + phi0)) * this_data')
                if m >= self.nfft / 2:
                    np.conj(demod[:, n], out=demod[:, n])
            else:
                demod[:, n] = wc * np.exp(sign * 1j * (2 * np.pi * foffs * t + phi0)) * data[:, n]
                if m >= self.nfft / 2:
                    demod[:, n] = np.conjugate(demod[:, n])
        return demod


class RoachBasebandWide10(RoachBasebandWide):

    def __init__(self, roach=None, wafer=0, roachip='roach', adc_valon=None):
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
                if (time.time() - t1) > timeout:
                    raise Exception("Connection timeout to roach")
                time.sleep(0.1)

        if adc_valon is None:
            import valon
            ports = valon.find_valons()
            if len(ports) == 0:
                self.adc_valon_port = None
                self.adc_valon = None
                print "Warning: No valon found!"
            else:
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

        self.adc_atten = -1
        self.dac_atten = -1
        self.bof_pid = None
        self.roachip = roachip
        try:
            self.fs = self.adc_valon.get_frequency_a()
        except:
            print "warning couldn't get valon frequency, assuming 512 MHz"
            self.fs = 512.0
        self.wafer = wafer
        self.dac_ns = 2 ** 16  # number of samples in the dac buffer
        self.raw_adc_ns = 2 ** 12  # number of samples in the raw ADC buffer
        self.nfft = 2 ** 10
        # self.boffile = 'bb2xpfb10mcr8_2013_Nov_18_0706.bof'
        self.boffile = 'bb2xpfb10mcr11_2014_Jan_20_1049.bof'
        self.bufname = 'ppout%d' % wafer
        self._window_mag = compute_window(npfb=2 * self.nfft, taps=2, wfunc=scipy.signal.flattop)

    def demodulate_data(self, data):
        """
        Demodulate the data from the FFT bin
        
        This function assumes that self.select_fft_bins was called to set up the necessary class attributes
        
        data : array of complex data
        
        returns : demodulated data in an array of the same shape and dtype as *data*
        """
        demod = np.zeros_like(data)
        t = np.arange(data.shape[0])
        for n, ich in enumerate(self.readout_selection):
            phi0 = self.phases[ich]
            k = self.tone_bins[ich]
            m = self.fft_bins[ich]
            if m >= self.nfft / 2:
                sign = 1.0
            else:
                sign = -1.0
            nfft = self.nfft
            ns = self.tone_nsamp
            foffs = (2 * k * nfft - m * ns) / float(ns)
            wc = self._window_response(foffs / 2) * (self.tone_nsamp / 2.0 ** 18)
            demod[:, n] = wc * np.exp(sign * 1j * (2 * np.pi * foffs * t + phi0)) * data[:, n]
            if m >= self.nfft / 2:
                demod[:, n] = np.conjugate(demod[:, n])
        return demod


class RoachHeterodyne(RoachInterface):

    def __init__(self, roach=None, wafer=0, roachip='roach', adc_valon=None):
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
                if (time.time() - t1) > timeout:
                    raise Exception("Connection timeout to roach")
                time.sleep(0.1)

        if adc_valon is None:
            import valon
            ports = valon.find_valons()
            if len(ports) == 0:
                self.adc_valon_port = None
                self.adc_valon = None
                print "Warning: No valon found!"
            else:
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

        self.adc_atten = -1
        self.dac_atten = -1
        self.bof_pid = None
        self.roachip = roachip
        try:
            self.fs = self.adc_valon.get_frequency_a()
        except:
            print "warning couldn't get valon frequency, assuming 512 MHz"
            self.fs = 512.0
        self.wafer = wafer
        self.dac_ns = 2 ** 16  # number of samples in the dac buffer
        self.raw_adc_ns = 2 ** 12  # number of samples in the raw ADC buffer
        self.nfft = 2 ** 14
        self.boffile = 'iq2xpfb14mcr4_2013_Aug_02_1446.bof'
        self.bufname = 'ppout%d' % wafer
        self._window_mag = compute_window(npfb=self.nfft, taps=2, wfunc=scipy.signal.flattop)

    def load_waveforms(self, i_wave, q_wave, fast=True):
        """
        Load waveforms for the two DACs
        
        i_wave,q_wave : arrays of 16-bit (dtype='i2') integers with waveforms for the two DACs
        
        fast : boolean
            decide what method for loading the dram 
        """
        data = np.zeros((2 * i_wave.shape[0],), dtype='>i2')
        data[0::4] = i_wave[::2]
        data[1::4] = i_wave[1::2]
        data[2::4] = q_wave[::2]
        data[3::4] = q_wave[1::2]
        self.r.write_int('dram_mask', data.shape[0] / 4 - 1)
        self._load_dram(data, fast=fast)

    def set_tone_freqs(self, freqs, nsamp, amps=None):
        """
        Set the stimulus tones to generate
        
        freqs : array of frequencies in MHz
            For Heterodyne system, these can be positive or negative to produce tones above and
            below the local oscillator frequency.
        nsamp : int, must be power of 2
            number of samples in the playback buffer. Frequency resolution will be fs/nsamp
        amps : optional array of floats, same length as freqs array
            specify the relative amplitude of each tone. Can set to zero to read out a portion
            of the spectrum with no stimulus tone.
        
        returns:
        actual_freqs : array of the actual frequencies after quantization based on nsamp
        """
        bins = np.round((freqs / self.fs) * nsamp).astype('int')
        actual_freqs = self.fs * bins / float(nsamp)
        bins[bins < 0] = nsamp + bins[bins < 0]
        self.set_tone_bins(bins, nsamp, amps=amps)
        self.fft_bins = self.calc_fft_bins(bins, nsamp)
        if self.fft_bins.shape[0] > 4:
            readout_selection = range(4)
        else:
            readout_selection = range(self.fft_bins.shape[0])

        self.select_fft_bins(readout_selection)
        return actual_freqs

    def set_tone_bins(self, bins, nsamp, amps=None):
        """
        Set the stimulus tones by specific integer bins
        
        bins : array of bins at which tones should be placed
        For Heterodyne system, negative frequencies should be placed in cannonical FFT order

        nsamp : int, must be power of 2
        number of samples in the playback buffer. Frequency resolution will be fs/nsamp

        amps : optional array of floats, same length as bins array
            specify the relative amplitude of each tone. Can set to zero to read out a portion
            of the spectrum with no stimulus tone.
        """
        spec = np.zeros((nsamp,), dtype='complex')
        self.tone_bins = bins.copy()
        self.tone_nsamp = nsamp
        phases = np.random.random(len(bins)) * 2 * np.pi
        self.phases = phases.copy()
        if amps is None:
            amps = 1.0
        self.amps = amps
        spec[bins] = amps * np.exp(1j * phases)
        wave = np.fft.ifft(spec)
        self.wavenorm = np.abs(wave).max()
        i_wave = np.round((wave.real / self.wavenorm) * (2 ** 15 - 1024)).astype('>i2')
        q_wave = np.round((wave.imag / self.wavenorm) * (2 ** 15 - 1024)).astype('>i2')
        self.i_wave = i_wave
        self.q_wave = q_wave
        self.load_waveforms(i_wave, q_wave)

    def calc_fft_bins(self, tone_bins, nsamp):
        """
        Calculate the FFT bins in which the tones will fall
        
        tone_bins: array of integers
            the tone bins (0 to nsamp - 1) which contain tones

        nsamp : length of the playback bufffer
        
        returns: fft_bins, array of integers.
        """
        tone_bins_per_fft_bin = nsamp / (self.nfft)
        fft_bins = np.round(tone_bins / float(tone_bins_per_fft_bin)).astype('int')
        return fft_bins

    def fft_bin_to_index(self, bins):
        """
        Convert FFT bins to FPGA indexes
        """
        idx = bins.copy()
        return idx

    def select_fft_bins(self, readout_selection):
        """
        Select which subset of the available FFT bins to read out
        
        Initially we can only read out from a subset of the FFT bins, so this function selects which bins to read out right now
        This also takes care of writing the selection to the FPGA with the appropriate tweaks
        
        The readout selection is stored to self.readout_selection
        The FPGA readout indexes is stored in self.fpga_fft_readout_indexes
        The bins that we are reading out is stored in self.readout_fft_bins
        
        readout_selection : array of ints
            indexes into the self.fft_bins array to specify the bins to read out
        """
        offset = 4
        idxs = self.fft_bin_to_index(self.fft_bins[readout_selection])
        order = idxs.argsort()
        idxs = idxs[order]
        self.readout_selection = np.array(readout_selection)[order]
        self.fpga_fft_readout_indexes = idxs
        self.readout_fft_bins = self.fft_bins[self.readout_selection]

        binsel = np.zeros((self.fpga_fft_readout_indexes.shape[0] + 1,), dtype='>i4')
        # evenodd = np.mod(self.fpga_fft_readout_indexes,2)
        #binsel[:-1] = np.mod(self.fpga_fft_readout_indexes/2-offset,self.nfft/2)
        #binsel[:-1] += evenodd*2**16
        binsel[:-1] = np.mod(self.fpga_fft_readout_indexes - offset, self.nfft)
        binsel[-1] = -1
        self.r.write('chans', binsel.tostring())

    def demodulate_data(self, data):
        """
        Demodulate the data from the FFT bin
        
        This function assumes that self.select_fft_bins was called to set up the necessary class attributes
        
        data : array of complex data
        
        returns : demodulated data in an array of the same shape and dtype as *data*
        """
        demod = np.zeros_like(data)
        t = np.arange(data.shape[0])
        for n, ich in enumerate(self.readout_selection):
            phi0 = self.phases[ich]
            k = self.tone_bins[ich]
            m = self.fft_bins[ich]
            if m >= self.nfft / 2:
                sign = -1.0
                doconj = True
            else:
                sign = -1.0
                doconj = False
            nfft = self.nfft
            ns = self.tone_nsamp
            foffs = (k * nfft - m * ns) / float(ns)
            demod[:, n] = np.exp(sign * 1j * (2 * np.pi * foffs * t + phi0)) * data[:, n]
            if doconj:
                demod[:, n] = np.conjugate(demod[:, n])
        return demod

    def get_data(self, nread=10, demod=True):
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
        print "getting data"
        bufname = 'ppout%d' % self.wafer
        chan_offset = 1
        draw, addr, ch = self._read_data(nread, bufname)
        if not np.all(ch == ch[0]):
            print "all channel registers not the same; this case not yet supported"
            return draw, addr, ch
        if not np.all(np.diff(addr) < 8192):
            print "address skip!"
        nch = self.readout_selection.shape[0]
        dout = draw.reshape((-1, nch))
        shift = np.flatnonzero(self.fpga_fft_readout_indexes / 2 == (ch[0] - chan_offset))[0] - (nch - 1)
        print shift
        dout = np.roll(dout, shift, axis=1)
        if demod:
            dout = self.demodulate_data(dout)
        return dout, addr

    def set_lo(self, lomhz=1200.0, chan_spacing=2.0):
        """
        Set the local oscillator frequency for the IQ mixers
        
        lomhz: float, frequency in MHz
        """
        self.adc_valon.set_frequency_b(lomhz, chan_spacing=chan_spacing)

    def set_dac_attenuator(self, attendb):
        if attendb < 0 or attendb > 63:
            raise ValueError("ADC Attenuator must be between 0 and 63 dB. Value given was: %s" % str(attendb))

        if attendb > 31.5:
            attena = 31.5
            attenb = attendb - attena
        else:
            attena = attendb
            attenb = 0
        self.set_attenuator(attena, le_bit=0x01)
        self.set_attenuator(attenb, le_bit=0x80)
        self.dac_atten = int(attendb * 2) / 2.0

    def set_adc_attenuator(self, attendb):
        if attendb < 0 or attendb > 31.5:
            raise ValueError("ADC Attenuator must be between 0 and 31.5 dB. Value given was: %s" % str(attendb))
        self.set_attenuator(attendb, le_bit=0x02)
        self.adc_atten = int(attendb * 2) / 2.0

    def _set_fs(self, fs, chan_spacing=2.0):
        """
        Set sampling frequency in MHz
        Note, this should generally not be called without also reprogramming the ROACH
        Use initialize() instead        
        """
        if self.adc_valon is None:
            print "Could not set Valon; none available"
            return
        self.adc_valon.set_frequency_a(fs, chan_spacing=chan_spacing)
        self.fs = fs


def test_sweep(ri):
    data = []
    tones = []
    ri.r.write_int('sync', 0)
    ri.r.write_int('sync', 1)
    ri.r.write_int('sync', 0)
    for k in range(ri.fft_bins.shape[0] / 4):
        ri.select_fft_bins(range(k * 4, (k + 1) * 4))
        time.sleep(0.1)
        d, addr = ri.get_data(2)
        data.append(d)
        tones.append(ri.tone_bins[ri.readout_selection])
    tones = np.concatenate(tones)
    order = tones.argsort()
    davg = np.concatenate([x.mean(0) for x in data])
    davg = davg[order]
    tones = tones[order]
    return tones, davg, data
