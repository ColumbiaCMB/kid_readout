import valon_synth
try:
    ver = valon_synth.valon_synth.__version__
    if float(ver) < 1.0:
        raise Exception('insufficient valon_synth version')
except AttributeError:
    raise Exception('insufficient valon_synth version')
from valon_synth import Synthesizer, SYNTH_A, SYNTH_B
import numpy as np
import os
import sys
import subprocess


def find_valons():
    """
    Find /dev/ttyUSB* ports with FTDI chips which are probably Valons
    
    Works by parsing the dmesg output and looking for lines containing FTDI (the brand of
    usb-rs232 chip used) and 'attached to' which will contain the ttyUSB* string
    
    returns a list of all unique ports that have FTDI chips.
    The list is sorted starting with the most recent entry in dmesg.
    Duplicate ports are ignored (only the most recent entry (line in dmesg) for that port is returned)
    """
    
    dmesg = check_output('dmesg | grep "FTDI.*attached to"',shell=True)
    lines = dmesg.split('\n')
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
    