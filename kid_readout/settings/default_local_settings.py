import socket

HOSTNAME = socket.gethostname()

if HOSTNAME == 'detectors':
    from kid_readout.settings.detectors_example import *
elif HOSTNAME == 'readout':
    from kid_readout.settings.readout_example import *

