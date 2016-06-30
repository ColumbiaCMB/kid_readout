import logging
import os
import sys
import time
import warnings
import socket
import subprocess

import numpy as np
import scipy
import zlib

import borph_utils
from kid_readout.roach.tests.mock_roach import MockRoach
from kid_readout.settings import BASE_DATA_DIR
from kid_readout.roach import tools
from kid_readout.measurement.core import StateDict
from kid_readout.measurement.basic import StreamArray
from kid_readout.measurement.misc import ADCSnap


CONFIG_FILE_NAME_TEMPLATE = os.path.join(BASE_DATA_DIR,'%s_config.npz')

logger = logging.getLogger(__name__)

class RoachInterface(object):
    """
    Base class for readout systems.

    These methods define an abstract interface that can be relied on to be consistent between the baseband and
    heterodyne readout systems.
    """

    def __init__(self, roach=None, roachip='roach', adc_valon=None, host_ip=None,
                 nfs_root='/srv/roach_boot/etch', lo_valon=None):
        """
        Abstract class to represent readout system

        roach: an FpgaClient instance for communicating with the ROACH.
                If not specified, will try to instantiate one connected to *roachip*
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
                the host_ip will be set appropriately based on the HOSTNAME.
        """
        self.is_roach2 = False
        self._using_mock_roach = False
        if roach:
            self.r = roach
            # Check if we're using a fake ROACH for testing. If so, disable additional externalities
            # This logic could be made more general if desired (i.e. has attribute mock
            #  or type name matches regex including 'mock'
            if type(roach) is MockRoach:
                self._using_mock_roach = True
        else:
            from corr.katcp_wrapper import FpgaClient
            logger.debug("Creating FpgaClient")
            self.r = FpgaClient(roachip)
            t1 = time.time()
            timeout = 10
            logger.debug("Waiting for connection to ROACH")
            while not self.r.is_connected():
                if (time.time() - t1) > timeout:
                    raise Exception("Connection timeout to roach")
                time.sleep(0.1)
            logger.debug("ROACH is connected")

        if adc_valon is None:
            from kid_readout.roach import valon
            ports = valon.find_valons()
            if len(ports) == 0:
                self.adc_valon_port = None
                self.adc_valon = None
                logger.warn("Warning: No valon found! You will not be able to change or verify the sampling frequency")
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
            from kid_readout.roach import valon
            self.adc_valon_port = adc_valon
            self.adc_valon = valon.Synthesizer(self.adc_valon_port)
        else:
            self.adc_valon = adc_valon

        if type(lo_valon) is str:
            from kid_readout.roach import valon
            self.lo_valon_port = lo_valon
            self.lo_valon = valon.Synthesizer(self.lo_valon_port)
        else:
            self.lo_valon = lo_valon

        if host_ip is None:
            hostname = socket.gethostname()
            if hostname == 'detectors':
                host_ip = '192.168.1.1'
            else:
                host_ip = '192.168.1.1'
        self.host_ip = host_ip
        self.roachip = roachip
        self.nfs_root = nfs_root
        self._config_file_name = CONFIG_FILE_NAME_TEMPLATE % self.roachip

        self.adc_atten = 31.5
        self.dac_atten = -1
        self.fft_gain = 0
        self.fft_bins = None
        self.tone_nsamp = None
        self.tone_bins = None
        self.phases = None
        self.amps = None
        self.readout_selection = None
        self.modulation_output = 0
        self.modulation_rate = 0
        self.wavenorm = None

        self.loopback = None
        self.debug_register = None

        # Things to be configured by subclasses
        self.lo_frequency = 0.0
        self.iq_delay = 0
        self.heterodyne = False
        self.bof_pid = None
        self.boffile = None
        self.wafer = None
        self.raw_adc_ns = 2 ** 12  # number of samples in the raw ADC buffer
        self.nfft = None
        # Boffile specific register names
        self._fpga_output_buffer = None


    def _general_setup(self):
        """
        Intended to be called after or at the end of subclass __init__
        """
        self._window_mag = tools.compute_window(npfb=2 * self.nfft, taps=2, wfunc=scipy.signal.flattop)
        try:
            self.hardware_delay_estimate = tools.boffile_delay_estimates[self.boffile]
        except KeyError:
            self.hardware_delay_estimate = tools.get_delay_estimate_for_nfft(self.nfft,self.heterodyne)

        try:
            self.fs = self.adc_valon.get_frequency_a()
        except:
            logger.warn("Couldn't get valon frequency, assuming 512 MHz")
            self.fs = 512.0
        self.bank = self.get_current_bank()

    # FPGA Functions
    def _update_bof_pid(self):
        if self.is_roach2:
            return
        if self.bof_pid:
            return
        if not self._using_mock_roach:
            try:
                self.bof_pid = borph_utils.get_bof_pid(self.roachip)
            except Exception, e:
                self.bof_pid = None
    #            raise e

    @property
    def num_tones(self):
        """
        Returns
        -------
        num_tones : int or None
            number of tones being played. None if unknown/not programmmed

        """
        if self.tone_bins is None:
            num_tones = None
        else:
            num_tones = self.tone_bins.shape[1] # We may want to update this later if some tones have
                                                 # zero tone_amplitude
        return num_tones

    @property
    def state_arrays(self):
        return self.get_state_arrays()

    def get_state_arrays(self):
        def copy_or_none(x):
            if x is None:
                return x
            else:
                return np.asanyarray(x).copy()
        state = StateDict(
                  tone_bin=copy_or_none(self.tone_bins),
                  tone_amplitude=copy_or_none(self.amps),
                  tone_phase=copy_or_none(self.phases),
                  tone_index=copy_or_none(self.readout_selection),
                  filterbank_bin=copy_or_none(self.fft_bins),
                  )
        return state

    @property
    def active_state_arrays(self):
        return self.get_active_state_arrays()

    def get_active_state_arrays(self):
        state = self.get_state_arrays()
        if state.tone_bin is not None:
            state.tone_bin = state.tone_bin[self.bank,:]
        if state.filterbank_bin is not None:
            state.filterbank_bin = state.filterbank_bin[self.bank,:]
        return state

    @property
    def state(self):
        return self.get_state()

    def get_state(self,include_registers=False):
        roach_state = StateDict(boffile=self.boffile,
                          heterodyne=self.heterodyne,
                          adc_sample_rate=self.fs*1e6, # roach still uses MHz, so convert to Hz
                          lo_frequency=self.lo_frequency*1e6, # roach still uses MHz, so convert to Hz
                          num_tones=self.num_tones,
                          modulation_rate=self.modulation_rate,
                          modulation_output=self.modulation_output,
                          waveform_normalization=self.wavenorm,
                          num_tone_samples=self.tone_nsamp,
                          num_filterbank_channels=self.nfft,
                          dac_attenuation=self.dac_atten,
                          bank=self.bank,
                          loopback=self.loopback,
                          debug_register=self.debug_register,
                          )
        if include_registers:
            for register in self.initial_values_for_writeable_registers:
                roach_state[register] = self.r.read_int(register)
        return roach_state


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

    def get_adc_measurement(self):
        epoch = time.time()
        s0, s1 = self.get_raw_adc()
        return ADCSnap(epoch=epoch,x=s0,y=s1,state=self.get_state())

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

    # TODO: this should raise a RoachError or return None if no bank is selected or the Roach isn't programmed.
    def get_current_bank(self):
        """
        Determine what tone bank the ROACH is currently set to use
        """
        try:
            bank_reg = self.r.read_int('dram_bank')
            mask_reg = self.r.read_int('dram_mask')
            if mask_reg == 0:
                self.bank = 0  # if mask is not set, bank is undefined, so call it 0
            else:
                self.bank = bank_reg / (mask_reg + 1)
        except RuntimeError:
            self.bank = 0  # this catches the case that the ROACH is not yet programmed
        return self.bank

    def select_bank(self, bank):
        if self.tone_nsamp is None:
            logger.warning("Attemped to select bank, but no tones have been loaded yet")
            self.bank = 0
            return
        dram_addr_per_bank = self.tone_nsamp / 2  # number of dram addresses per bank
        mask_reg = dram_addr_per_bank - 1
        bank_reg = dram_addr_per_bank * bank
        self._pause_dram()
        self.r.write_int('dram_bank', bank_reg)
        self.r.write_int('dram_mask', mask_reg)
        self._unpause_dram()
        self.bank = bank
#        try:
#            self.select_fft_bins(self.readout_selection)
#        except AttributeError:
#            self.select_fft_bins(np.arange(self.tone_bins.shape[1]))


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

    def set_loopback(self,enable):
        raise NotImplementedError("Must be implemented by subclasses")

    def save_state(self):
        if self._using_mock_roach:
            return #don't save anything when using mock
        np.savez(self._config_file_name,
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
                 modulation_output=self.modulation_output,
                 lo_frequency=self.lo_frequency,
                 iq_delay=self.iq_delay)
        try:
            os.chmod(self._config_file_name, 0777)
        except:
            pass

    def initialize(self, fs=512.0, start_udp=True, use_config=True, raise_if_not_locked=True):
        """
        Reprogram the ROACH and get things running

        Parameters
        ----------
        fs: float
            Sampling frequency in MHz
        start_udp
        use_config
        raise_if_not_locked

        Returns
        -------
        reprogrammed: bool
            True if the ROACH was reprogrammed

        """
        reprogrammed = False
        if use_config:
            try:
                state = np.load(self._config_file_name)
                logger.info("Loaded ROACH state from %s", self._config_file_name)
            except IOError:
                logger.info("Could not load previous roach state")
                state = None
        else:
            state = None
        if state is not None:
            try:
                crc = self.r.read_int('sys_scratchpad')
                logger.debug("Programmed boffile crc: %d" % crc)
            except RuntimeError:
                logger.debug("Could not read scratchpad, ROACH probably not configured yet")
                crc = None

            try:
                self._update_bof_pid()
            except Exception:
                self.bof_pid = None
            if (crc != zlib.crc32(self.boffile) or
                not self.is_roach2 and (self.bof_pid is None or self.bof_pid != state['bof_pid'])):
                logger.debug("ROACH configuration does not match saved state")
                state = None
        boffile_mismatch = False
        if state is not None:
            try:
                boffile_mismatch = state['boffile'] != self.boffile
            except KeyError:
                boffile_mismatch = True
        if state is None or boffile_mismatch:
            reprogrammed = True
            logger.info("Reinitializing system")
            self._set_fs(fs)
            logger.debug("Deprogramming")
            try:
                self.r.progdev('')
            except RuntimeError, e:
                pass
            logger.info("Programming %s", self.boffile)
            self.r.progdev(self.boffile)
            try:
                self.r.write_int('sys_scratchpad',zlib.crc32(self.boffile))
            except RuntimeError, e:
                logger.exception("Unable to write to ROACH scratchpad register. Something is very wrong")
                raise RuntimeError("Unable to write to ROACH scratchpad register. Something is very wrong")
            self.bof_pid = None
            self._update_bof_pid()
            self.set_fft_gain(4)
            self.r.write_int('dacctrl', 2)
            self.r.write_int('dacctrl', 1)
            estfs = self.measure_fs()
            if np.abs(fs - estfs) > 2.0:
                logger.error("FPGA clock may not be locked to sampling clock!")
                if raise_if_not_locked:
                    raise RuntimeError("ROACH not locked to Valon: Requested sampling rate %.1f MHz. Estimated sampling rate %.1f MHz" % (fs,estfs))
            logger.info("Requested sampling rate %.1f MHz. Estimated sampling rate %.1f MHz" % (fs, estfs))
            if start_udp and not self._using_mock_roach:
                logger.debug("starting udp server process on PPC")
                borph_utils.start_server(self.bof_pid, self.roachip)

            self.set_loopback(False)
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
            self.lo_frequency = state['lo_frequency'][()]
            self.iq_delay = state['iq_delay'][()]
        self.set_debug(0) # Turn off debug and loopback no matter what to avoid surprises
        self.set_loopback(False)

        return reprogrammed

    def measure_fs(self):
        """
        Estimate the sampling rate

        This takes about 2 seconds to run
        returns: fs, the approximate sampling rate in MHz
        """
        return 2 * self.r.est_brd_clk()

    def measure_hardware_delay(self,**kwargs):
        return tools.measure_hardware_delay(self,**kwargs)

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
        logger.info("Writing %.1f kB to DRAM", (nbytes/2.**10))
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
        logger.debug("bank_offset= %d  start_offset=%d  start_offset_bytes=%d", bank_offset , start_offset,
                    start_offset_bytes)
        for bank in range(nbanks):
            logger.debug("writing DRAM bank %d", (bank + bank_offset))
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
                logger.debug("failure writing to dram, trying again", exc_info=True)
            tries = tries - 1
        raise Exception("Writing to dram failed!")

    def _load_dram_ssh(self, data, offset_bytes=0, datafile='boffiles/dram.bin'):
        offset_blocks = offset_bytes / 512  #dd uses blocks of 512 bytes by default
        self._update_bof_pid()
        self._pause_dram()
        if self._using_mock_roach:
            time.sleep(0.01) #TODO: Can make this take a realistic amount of time if desired
        else:
            data.tofile(os.path.join(self.nfs_root, datafile))
            dram_file = '/proc/%d/hw/ioreg/dram_memory' % self.bof_pid
            datafile = '/' + datafile
            # TODO: Verify that this change is fine.
            # This was using borph_utils.check_output(), which seems to be the same as subprocess.check_output().
            # Capture stderr to stdout because dd prints to stderr.
            command = 'ssh root@%s "dd seek=%d if=%s of=%s"' % (self.roachip, offset_blocks, datafile, dram_file)
            result = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
            logger.debug(result)
        self._unpause_dram()

    def _sync(self,loopback=None):
        if loopback is not None:
            warnings.warn("loopback parameter to _sync is deprecated, use set_loopback method")
            self.set_loopback(loopback)
        if self.loopback:
            base_value = 2
        else:
            base_value = 0
        self.r.write_int('sync', 0+base_value)
        self.r.write_int('sync', 1+base_value)
        self.r.write_int('sync', 0+base_value)

    ### Other hardware functions (attenuator, valon)
    def set_attenuator(self, attendb, gpio_reg='gpioa', data_bit=0x08, clk_bit=0x04, le_bit=0x02):
        atten = int(attendb * 2)
        try:
            self.r.write_int(gpio_reg, 0x00)
        except RuntimeError:
            raise RuntimeError("ROACH not programmed, cannot set attenuators")
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
        raise NotImplementedError("ADC attenuator is no longer adjustable.")

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

    def _set_fs(self, fs, chan_spacing = 2.0):
        """
        Set sampling frequency in MHz
        Note, this should generally not be called without also reprogramming the ROACH
        Use initialize() instead
        """
        if np.mod(fs,chan_spacing) > 1e-7:
            raise ValueError("The requested sampling frequency %f is not divisible by the channel spacing %f" % (fs,
                                                                                                                 chan_spacing))
        if self.adc_valon is None:
            logger.warning("Could not set Valon; none available")
            return
        self.adc_valon.set_frequency_a(fs, chan_spacing=chan_spacing)  # for now the baseband readout uses both valon
        #  outputs,
        self.fs = float(fs)

    def set_debug(self,value):
        self.r.write_int('debug',value)
        self.debug_register = value

    @property
    def blocks_per_second(self):
        return self.blocks_per_second_per_channel*len(self.readout_selection)

    @property
    def blocks_per_second_per_channel(self):
        raise NotImplementedError("blocks_per_second needs to be implemented for this subclass")

    def get_measurement(self, num_seconds, power_of_two=True, demod=True, **kwargs):
        num_blocks = self.blocks_per_second*num_seconds
        if num_blocks == 0:
            num_blocks = 1 # we have to get at least one block
        if power_of_two:
            log2 = np.int(np.round(np.log2(num_blocks)))  # Changed to int so that num_blocks is an int
            if log2 < 0:
                log2 = 0
            num_blocks = 2 ** log2
        return self.get_measurement_blocks(num_blocks, demod=demod, **kwargs)

    def get_measurement_blocks(self, num_blocks, demod=True, **kwargs):
        epoch = time.time()  # This will be improved
        data, seqnos = self.get_data(num_blocks, demod=demod)
        sequence_start_number = int(seqnos[0])  # The numpy datatype causes IO problems.
        if np.isscalar(self.amps):
            tone_amplitude = self.amps * np.ones(self.tone_bins.shape[1], dtype='float')
        else:
            tone_amplitude = self.amps.copy()
        output_order = self.readout_selection.argsort()
        measurement = StreamArray(tone_bin=self.tone_bins[self.bank, :].copy(),
                                  tone_amplitude=tone_amplitude,  # already copied
                                  tone_phase=self.phases.copy(),
                                  tone_index=self.readout_selection.copy()[output_order],
                                  filterbank_bin=self.fft_bins[self.bank, self.readout_selection].copy()[output_order],
                                  epoch=epoch,
                                  sequence_start_number=sequence_start_number,
                                  s21_raw=data[:,output_order].T,  # transpose for now, because measurements are
                                          # organized channel,time
                                  data_demodulated=demod,
                                  roach_state=self.get_state(),
                                  **kwargs)
        return measurement

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
            logger.error("read only partway because of error:", exc_info=True)
        tot = time.time() - tic
        logger.debug("read %d in %.1f seconds, %.2f samples per second, idle %.2f per read" % (
                        nread, tot, (nread * 2 ** 12 / tot), idle / (nread * 1.0)))
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
                    logger.error("read only partway because of error:", exc_info=True)
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
        logger.debug("read %d in %.1f seconds, %.2f samples per second, idle %.2f per read" % (
                        n, tot, (n * 2 ** 12 / tot), idle / (n * 1.0)))


class RoachError(Exception):
    """
    This class is raised on Roach-specific errors.
    """
    pass
