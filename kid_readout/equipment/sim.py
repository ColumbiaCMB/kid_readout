"""
This module contains classes to interface with SRS SIM hardware.

The classes do not maintain any internal state that corresponds to hardware settings: all state queries are directed
to the hardware.

As described below in more detail, the query commands return strings, booleans, or floats.

Settings such as excitation codes that take only integer values accept either an integer or a string corresponding to
an integer; they return a string corresponding to an integer.

Many of the SIM commands accept either an integer or a token. For example, the SIM921 excitation mode can be set to
constant current using either 'MODE CURRENT' or 'MODE 2'. These commands accept a valid token, a string corresponding
to an integer, or an integer. Depending on the token mode, they return a string that is either a token or represents
an integer.

Settings that can be only 'OFF' or 'ON' are fundamentally boolean. The corresponding commands accept strings ('OFF',
'0') or ('ON', '1'), or booleans. These commands always return True or False, not a string, regardless of the token
mode.

Commands that return values measured by an instrument, such as a resistance, return floats.

Commands that accept tokens are case-insensitive. (However, note that calibration curve identification strings are
converted to uppercase.)
"""
from __future__ import division
import os
import time
import serial
import numpy as np
from collections import OrderedDict


class SIMError(Exception):
    pass


class SIMValueError(SIMError):
    pass


class SIMTimeout(SIMError):
    pass


class SIM(object):

    termination = '\n'

    token_to_boolean = {'OFF': False,
                        '0': False,
                        'ON': True,
                        '1': True}

    boolean_to_token = {True: 'ON',
                        False: 'OFF'}

    def __init__(self, serial, parent_and_port=(None, None)):
        self.serial = serial
        self.parent = parent_and_port[0]
        self.port = parent_and_port[1]

    def send(self, message):
        if self.parent is not None:
            self.parent.connect(self.port)
        self.serial.write(message + self.termination)
        if self.parent is not None:
            self.parent.disconnect()

    # Consider raising SIMTimeout for blank messages;
    # needs to handle disconnection elegantly.
    def receive(self):
        if self.parent is not None:
            self.parent.connect(self.port)
        response = self.serial.readline().strip()
        if self.parent is not None:
            self.parent.disconnect()
        return response

    def send_and_receive(self, message):
        if self.parent is not None:
            self.parent.connect(self.port)
        self.serial.write(message + self.termination)
        response = self.serial.readline().strip()
        if self.parent is not None:
            self.parent.disconnect()
        return response

    # Handle boolean user input for commands that accept ('OFF', '0', 'ON', '1').
    def _boolean_input(self, thing):
        try:
            return self.boolean_to_token[token_to_boolean.get(str(thing).upper(), thing)]
        except KeyError:
            raise SIMValueError("Invalid boolean setting {}".format(thing))

    # Handle boolean output for commands that return ('OFF', '0', 'ON', '1').
    def _boolean_output(self, thing):
        return self.token_to_boolean[thing]

    @property
    def token(self):
        """
        The token mode.

        This property implements the TOKN(?) command.

        When True, the SIM returns tokens such as 'ON' or 'VOLTAGE'; when False, the SIM returns integer codes instead.
        """
        return self._boolean_output(self.send_and_receive('TOKN?'))

    @token.setter
    def token(self, boolean):
        self.send('TOKN {}'.format(self._boolean_input([boolean])))

    @property
    def identification(self):
        return self.send_and_receive('*IDN?')

    def reset(self):
        self.send('*RST')

    def clear_status(self):
        """
        Clear all status registers. The registers cleared vary by device.

        This method implements the *CLS command.
        """
        self.send('*CLS')


class SIM900(SIM):

    # This is the ASCII <ESC> character.
    escape = chr(27)

    def __init__(self, serial_port, baudrate=9600, timeout=2, autodetect=True):
        self.serial = serial.Serial(port=serial_port, baudrate=baudrate, timeout=timeout)
        self.ports = OrderedDict([('1', None),
                                  ('2', None),
                                  ('3', None),
                                  ('4', None),
                                  ('5', None),
                                  ('6', None),
                                  ('7', None),
                                  ('8', None),
                                  ('9', None),
                                  ('A', None),
                                  ('B', None),
                                  ('C', None),
                                  ('D', None)])
        self.parent = None
        self.disconnect()
        self.reset()
        self.flush()
        self.SIM_reset()
        if autodetect:
            self.autodetect()

    def broadcast(self, message):
        self.send('BRDT "{}"'.format(message))

    def parse_definite_length(self, message):
        length_bytes = int(message[1])
        return message[2 + length_bytes:]

    def parse_message(self, message):
        """
        Parse a message of the form
        'MSG 1,something'
        and return the port and message:
        ('1', 'something')
        """
        header, content = message.split(',', 1)
        port = header[-1]
        return port, content

    def flush(self, port=None):
        """
        Flush the SIM900 input and output queue for the given port, or for all ports if no port is given.

        This method implements the FLSH command.
        """
        if port is None:
            self.send('FLSH')
        elif str(port) in self.ports:
            self.send('FLSH {}'.format(port))
        else:
            raise SIMValueError("Invalid port {}'.format(port)")

    def SIM_reset(self, port=None):
        """
        Send the SIM reset signal to the given SIM port, meaning port
        1 through port 8, or to all SIM ports if no port is specified.

        This method implements the SRST command.
        """
        if port is None:
            self.send('SRST')
        elif int(port) in range(1, 9):
            self.send('SRST {}'.format(int(port)))
        else:
            raise SIMValueError("Invalid port {}".format(port))

    def connect(self, port):
        if self.connected is not None:
            raise SIMError("Connected to port {}".format(self.connected))
        self.send('CONN {}, "{}"'.format(port, self.escape))
        self.connected = port

    def disconnect(self):
        self.send(self.escape)
        self.connected = None

    def autodetect(self):
        # Upgrade with methods when available.
        self.send('BRER 510') # Turn on broadcasting for ports 1-8
        self.send('RPER 510') # Turn on pass-through for ports 1-8
        self.broadcast('*IDN?') # Ask everything for its identification
        self.send('WAIT 1000') # Wait for one second
        self.send('BRER 0') # Turn off broadcasting
        self.send('RPER 0') # Turn off message pass-through. Check that this keeps self-sent messages on!
        lines = [line.strip() for line in self.serial.readlines()]
        lines = [line for line in lines if line] # Remove blank messages
        for line in lines:
            port, message = self.parse_message(line)
            SRS, sim, serial_number, firmware_version = self.parse_definite_length(message).split(',')
            try:
                self.ports[port] = globals()[sim](self.serial, (self, port)) # Update
            except KeyError as e:
                self.ports[port] = str(e)


class SIMThermometer(SIM):
    """
    This is intended to be an abstract class that allows the
    temperature sensors to share code.

    For the SIM921 resistance bridge, the parameter number refers to
    calibration curve 1, 2, or 3. For the SIM922 diode temperature
    monitor, the parameter number refers to diode channel 1, 2, 3, or
    4; there can be only one user calibration curve stored per channel.
    """

    # This is the maximum fractional error used by the
    # validate_curve() method. It allows for a small rounding error
    # due to limited storage space in the SIM.  I have seen fractional
    # errors of up to 1.2e-5 or so.
    maximum_fractional_error = 1e-4

    # Consider adding enumeration of allowable curve numbers.
    def curve_info(self, number):
        message = self.send_and_receive('CINI? {}'.format(number)).split(',')
        format = message[0]
        identification = message[1]
        points = int(message[2])
        return format, identification, points

    def initialize_curve(self, number, format, identification):
        self.send('CINI {}, {}, {}'.format(number, format, identification))

    def read_curve(self, number):
        format, identification, points = self.curve_info(number)
        sensor = []
        temperature = []
        # The indexing is one-based.
        for n in range(1, points + 1):
            # The SIM921 separator is a comma, as its manual says, but
            # the SIM922 separator is a space and its manual lies.
            message = self.send_and_receive('CAPT? {}, {}'.format(number, n)).split(self.CAPT_separator)
            sensor.append(float(message[0]))
            temperature.append(float(message[1]))
        return CalibrationCurve(sensor, temperature, identification, format)

    def write_curve(self, number, curve):
        if curve.sensor.size > self.maximum_temperature_points:
            raise SIMError("Curve contains too many points.")
        self.initialize_curve(number, curve.format, curve.identification)
        if self.parent is not None:
            self.parent.connect(self.port)
        for n in range(curve.sensor.size):
            self.serial.write('CAPT {}, {}, {}{}'.format(number, curve.sensor[n], curve.temperature[n], self.termination))
            time.sleep(self.write_delay)
        if self.parent is not None:
            self.parent.disconnect()
        if not self.validate_curve(number, curve):
            raise SIMError("Curve data was not written correctly.")

    def validate_curve(self, number, curve):
        format, identification, points = self.curve_info(number)
        stored = self.read_curve(number)
        # If the writing speed is too fast some points may be skipped,
        # which will cause the array comparisons below to raise a
        # ValueError.
        try:
            return (np.all(abs(stored.sensor / curve.sensor - 1) <
                           self.maximum_fractional_error) and
                    np.all(abs(stored.temperature / curve.temperature - 1) <
                           self.maximum_fractional_error) and
                    (stored.identification == curve.identification) and
                    (stored.format == curve.format))
        except ValueError:
            return False


class SIM921(SIMThermometer):

    # The documentation for the CAPT command is correct.
    CAPT_separator = ','

    # The manual doesn't mention the maximum number of points per
    # curve, but with the other system we ran into problems when using
    # more points.
    maximum_temperature_points = 225

    # This is in seconds; points were occasionally dropped at 0.1 seconds.
    write_delay = 0.5

    # Minimum and maximum excitation frequencies in Hz:
    minimum_frequency = 1.95
    maximum_frequency = 61.1

    # Excitation commands

    @property
    def frequency(self):
        """
        The excitation frequency in Hz.

        This property implements the FREQ(?) command.
        """
        return float(self.send_and_receive('FREQ?'))

    @frequency.setter
    def frequency(self, frequency):
        if not self.minimum_frequency <= frequency <= self.maximum_frequency:
            raise SIMValueError("Valid excitation frequency range is from {} to {} Hz".format(self.minimum_frequency, self.maximum_frequency))
        self.send('FREQ {}'.format(frequency))

    @property
    def range(self):
        """
        The resistance range code. See the manual for code meanings.

        This property implements the RANG(?) command.
        """
        return int(self.send_and_receive('RANG?'))

    @range.setter
    def range(self, code):
        if not int(code) in range(10):
            raise SIMValueError("Valid range codes are integers 0 through 9.")
        self.send('RANG {}'.format(int(code)))

    @property
    def excitation(self):
        """
        The voltage excitation code. See the manual for code meanings.

        This property implements the EXCI(?) command.
        """
        return int(self.send_and_receive('EXCI?'))

    @excitation.setter
    def excitation(self, code):
        if not int(code) in range(-1, 9):
            raise SIMValueError("Valid excitation codes are integers -1 through 8.")
        self.send('EXCI {}'.format(int(code)))

    @property
    def excitation_on(self):
        """
        The excitation state.

        This property implements the EXON(?) command.
        """
        return self._boolean_output(self.send_and_receive('EXON?'))

    @excitation_on.setter
    def excitation_on(self, boolean):
        self.send('EXON {}'.format(self._boolean_input(boolean)))

    @property
    def excitation_mode(self):
        """
        The excitation mode.

        This property implements the MODE(?) command.
        """
        return self.send_and_receive('MODE?')

    @excitation_mode.setter
    def excitation_mode(self, mode):
        if not str(mode).upper() in ('PASSIVE', '0', 'CURRENT', '1', 'VOLTAGE', '2', 'POWER', '3'):
            raise SIMValueError("Invalid excitation mode.")
        self.send('MODE {}'.format(mode))

    @property
    def excitation_current(self):
        """
        The actual excitation current amplitude, in amperes.

        This property implements the IEXC? command.
        """
        return float(self.send_and_receive('IEXC?'))

    @property
    def excitation_voltage(self):
        """
        The actual excitation voltage amplitude, in volts.

        This property implements the VEXC? command.
        """
        return float(self.send_and_receive('VEXC?'))

    # Measurement commands

    @property
    def resistance(self):
        """
        Return the measured resistance.

        This property implements the RVAL? command.

        Multiple measurements and streaming are not yet implemented.
        """
        return float(self.send_and_receive('RVAL?'))

    @property
    def resistance_deviation(self):
        """
        The resistance deviation, in ohms, from the setpoint.

        This property implements the RDEV? command.

        Multiple measurements and streaming are not yet implemented.
        """
        return float(self.send_and_receive('RDEV?'))

    @property
    def temperature(self):
        """
        Return the temperature calculated using the current calibration curve.

        This property implements the TVAL? command.

        Multiple measurements and streaming are not yet implemented.
        """
        return float(self.send_and_receive('TVAL?'))

    @property
    def temperature_deviation(self):
        """
        The temperature deviation, in ohms, from the setpoint.

        This property implements the TDEV? command.

        Multiple measurements and streaming are not yet implemented.
        """
        return float(self.send_and_receive('TDEV?'))

    # The PHAS? command is not yet implemented.

    # The TPER(?) command is not yet implemented.

    # The SOUT command is not yet implemented.

    @property
    def display(self):
        """
        The display state. See the manual for meanings.

        This property implements the DISP(?) command.

        Only the range codes are implemented, not the string values.
        """
        return int(self.send_and_receive('DISP?'))

    @display.setter
    def display(self, code):
        if not int(code) in range(9):
            raise SIMValueError("Valid display codes are integers 0 through 8.")
        self.send('DISP {}'.format(int(code)))

    # Post-detection processing commands.

    def filter_reset(self):
        """
        Reset the post-detection filter.

        This method implements the FRST command.
        """
        self.send('FRST')

    @property
    def time_constant(self):
        """
        The filter time constant code. See the manual for meanings.

        This property implements the TCON(?) command.
        """
        return int(self.send_and_receive('TCON?'))

    @time_constant.setter
    def time_constant(self, code):
        if not int(code) in range(-1, 7):
            raise SIMValueError("Valid time constant codes are integers -1 through 6.")
        self.send('TCON {}'.format(int(code)))

    @property
    def phase_hold(self):
        """
        The phase hold state.

        This property implements the PHLD command.
        """
        return self._boolean_output(self.send_and_receive('PHLD?'))

    @phase_hold.setter
    def phase_hold(self, boolean):
        self.send('PHLD {}'.format(self._boolean_input(boolean)))

    # Calibration curve commands

    @property
    def display_temperature(self):
        """
        The temperature display mode.

        This property implements the DTEM(?) command.
        """
        return self._boolean_output(self.send_and_receive('DTEM?'))

    @display_temperature.setter
    def display_temperature(self, boolean):
        self.send('DTEM {}'.format(self._boolean_input(boolean)))

    @property
    def analog_output_temperature(self):
        """
        The analog output mode.

        This property implements the ATEM(?) command.
        """
        return self._boolean_output(self.send_and_receive('ATEM?'))

    @analog_output_temperature.setter
    def analog_output_temperature(self, boolean):
        self.send('ATEM {}'.format(self._boolean_input(boolean)))

    @property
    def active_curve(self):
        """
        The number of the active calibration curve: 1, 2, or 3.

        This property implements the CURV(?) command.
        """
        return self.send_and_receive('CURV?')

    @active_curve.setter
    def active_curve(self, number):
        if not str(number) in ('1', '2', '3'):
            raise SIMValueError("Curve number must be 1, 2, or 3.")
        self.send('CURV {}'.format(number))

    # The CINI(?) and CAPT(?) commands are implemented in SIMThermometer.

    # Autoranging commands

    def autorange_gain(self):
        """
        Perform a gain autorange cycle, which should take about two seconds.

        This method implements the AGAI(?) command.

        This method will return only when the autorange cycle completes.
        """
        self.send('AGAI ON')
        while self._boolean_output(self.send_and_receive('AGAI?')):
            time.sleep(0.5)

    @property
    def autorange_display(self):
        """
        The display autorange mode.

        This property implements the ADIS(?) command.
        """
        return self._boolean_output(self.send_and_receive('ADIS?'))

    @autorange_display.setter
    def autorange_display(self, boolean):
        self.send('ADIS {}'.format(self._boolean_input(boolean)))

    def autocalibrate(self):
        """
        Initiate the internal autocalibration cycle.
        """
        self.send('ACAL')

    # Setpoint and analog output commands.
    # Not yet implemented.

    # Interface commands.
    # The *IDN, *RST, and TOKN commands are implemented in SIM.

    # Status commands.


    # The *CLS command is implemented in SIM.


class SIM922(SIMThermometer):

    # The documentation for the CAPT command is incorrect: the
    # separator is a space, not a comma.
    CAPT_separator = ' '

    # The manual says that this is the maximum number of points per
    # channel, but I haven't checked it yet.
    maximum_temperature_points = 256

    # This is in seconds; points were sometimes dropped at 0.5 seconds and below.
    write_delay = 1

    def voltage(self, channel):
        return float(self.send_and_receive('VOLT? {}'.format(channel)))

    def temperature(self, channel):
        return float(self.send_and_receive('TVAL? {}'.format(channel)))

    def get_curve_type(self, channel):
        return self.send_and_receive('CURV? {}'.format(channel))

    def set_curve_type(self, channel, curve_type):
        if not str(curve_type).upper() in ('0', 'STAN', '1', 'USER'):
            raise SIMValueError("Invalid curve type.")
        self.send('CURV {}, {}'.format(channel, curve_type))


class SIM925(SIM):
        pass


class CalibrationCurve(object):

    def __init__(self, sensor, temperature, identification, format='0'):
        """
        This class represents and calibration curve.

        This class stores identification strings as uppercase because
        the hardware stores them that way. This allows the
        validate_curve() method to work.
        """
        self.sensor = np.array(sensor)
        if not all(np.diff(self.sensor)):
            raise SIMError("Sensor values must increase monotonically.")
        self.temperature = np.array(temperature)
        if not self.sensor.size == self.temperature.size:
            raise SIMError("Different numbers of sensor and temperature points.")
        self.identification = str(identification).upper()
        self.format = str(format)


def load_curve(filename, format='0'):
    identification = os.path.splitext(os.path.basename(filename))[0]
    sensor, temperature = np.loadtxt(filename, unpack=True)
    return CalibrationCurve(sensor, temperature, identification, format)


def save_curve(directory, curve, format=('%.5f\t%.5f'), newline='\r\n', extension='.txt'):
    filename = os.path.join(directory, curve.identification + extension)
    columns = np.empty((curve.sensor.size, 2))
    columns[:, 0] = curve.sensor
    columns[:, 1] = curve.temperature
    np.savetxt(filename, columns, fmt=format, newline=newline)
    return filename
