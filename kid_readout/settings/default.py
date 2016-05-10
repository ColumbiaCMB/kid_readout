"""
This is the default settings module.

The values of any variables that might differ between deploys of the package should either be None or should be
calculated at runtime. This file is under version control so it must work across all deploys. All of the tests should
pass with no local settings overriding the values here.

What variables should be included in this file?

Any settings that are used by library code (i.e. not scripts) to run, especially if test code needs to check whether
some variable is None to determine whether corresponding hardware is present. On the other hand, if you need a hardware
setting just to run a script, it's better not add a None default here.
"""
import os
import socket


# TODO: move away from allowing HOSTNAME to determine code paths in analysis; for data collection, use CRYOSTAT.
HOSTNAME = socket.gethostname()

# This is the directory into which data will be written. The default should always exist so that test code can use it.
if os.path.exists(os.path.join('/data', socket.gethostname())):
    BASE_DATA_DIR = os.path.join('/data', socket.gethostname())
else:
    BASE_DATA_DIR = '/tmp'

# The name of the cryostat, if any.
CRYOSTAT = None

# The path to the directory containing temperature log files.
TEMPERATURE_LOG_DIR = None

# ROACH
ROACH_HOST_IP = None
ROACH_IS_HETERODYNE = None
ROACH1_VALON = None
ROACH1_IP = None
ROACH2_VALON = None
ROACH2_IP = None
MARK2_VALON = None
