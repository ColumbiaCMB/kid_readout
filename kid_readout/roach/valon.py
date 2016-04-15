import valon_synth
import warnings
try:
    ver = valon_synth.valon_synth.__version__
    if float(ver) < 1.0:
        warnings.warn('insufficient valon_synth version')
except AttributeError:
    warnings.warn('insufficient valon_synth version')
from valon_synth import Synthesizer, SYNTH_A, SYNTH_B
import numpy as np
import os
import sys
import subprocess
import re

VALON_SERIAL_NUMBERS = dict(roach='AM01H05A',
                            roach2='A101FK1K',
                            mark2='A101FK1H')

usbre = re.compile(r"usb (?P<port>\d*-\d*)")


def find_valons():
    by_id = find_valons_by_id()
    if by_id is not None:
        return [by_id]
    print "couldn't find valons by id, trying dmesg"
    return find_valons_with_dmesg()

def find_valons_by_id():
    basepath = '/dev/serial/by-id/'
    serial_ports = os.listdir(basepath)
    for port in serial_ports:
        if port.find('usb-FTDI_FT232R_USB_UART') >= 0:
            return os.path.realpath(os.path.join(basepath,port))
    return None

def find_valons_with_dmesg():
    """
    Find /dev/ttyUSB* ports with FTDI chips which are probably Valons
    
    Works by parsing the dmesg output and looking for lines containing FTDI (the brand of
    usb-rs232 chip used) and 'attached to' which will contain the ttyUSB* string
    
    returns a list of all unique ports that have FTDI chips.
    The list is sorted starting with the most recent entry in dmesg.
    Duplicate ports are ignored (only the most recent entry (line in dmesg) for that port is returned)
    """
    
    try:
        dmesg = check_output('dmesg | grep "FT232RL"',shell=True)
    except subprocess.CalledProcessError:
        # grep failed so no ports found
        return []
    lines = dmesg.split('\n')
    lines = [x for x in lines if len(x) > 0]
    m = usbre.search(lines[-1])
    usbport = m.group('port')
    try:
        dmesg = check_output(('dmesg | grep "usb %s.*now attached to"' % usbport),shell=True)
    except subprocess.CalledProcessError:
        # grep failed so no ports found
        return []
    lines = dmesg.split('\n')
    lines = [x for x in lines if len(x) > 0]
    lines = lines[-1:]
    ports = []
    for ln in lines[::-1]:
        idx = ln.find('ttyUSB')
        if idx >= 0:
            port = '/dev/' + ln[idx:]
            if port not in ports:
                ports.append(port)
    return ports
    
def check_output(*popenargs, **kwargs):
    r"""Run command with arguments and return its output as a byte string.

    If the exit code was non-zero it raises a CalledProcessError.  The
    CalledProcessError object will have the return code in the returncode
    attribute and output in the output attribute.

    The arguments are the same as for the Popen constructor.  Example:

    >>> check_output(["ls", "-l", "/dev/null"])
    'crw-rw-rw- 1 root root 1, 3 Oct 18  2007 /dev/null\n'

    The stdout argument is not allowed as it is used internally.
    To capture standard error in the result, use stderr=STDOUT.

    >>> check_output(["/bin/sh", "-c",
    ...               "ls -l non_existent_file ; exit 0"],
    ...              stderr=STDOUT)
    'ls: non_existent_file: No such file or directory\n'
    
    copied from python2.7
    """
    if 'stdout' in kwargs:
        raise ValueError('stdout argument not allowed, it will be overridden.')
    process = subprocess.Popen(stdout=subprocess.PIPE, *popenargs, **kwargs)
    output, unused_err = process.communicate()
    retcode = process.poll()
    if retcode:
        cmd = kwargs.get("args")
        if cmd is None:
            cmd = popenargs[0]
        raise subprocess.CalledProcessError(retcode, cmd)#, output=output)
    return output
    