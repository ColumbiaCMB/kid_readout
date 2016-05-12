"""
This file tries to use sensible settings depending on the HOSTNAME
"""
import socket

HOSTNAME = socket.gethostname()

if HOSTNAME == 'detectors':
    from kid_readout.settings._detectors import *
elif HOSTNAME == 'readout':
    from kid_readout.settings._readout import *

