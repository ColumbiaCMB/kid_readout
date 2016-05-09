"""
This is the global settings module for the package.

It should contain values that are constant across systems and code to determine values that can be discovered at
runtime. The values of any variables that might vary across deploys of the package should be None here.

This module imports all settings from local_settings.py, which is listed in .gitignore and is thus not under version
control. The local settings override the settings in this file. Values that vary between deploys should be imported
through the local settings.

What variables should be included in this file?
Any settings that are used by library code (i.e. not scripts) to run, especially if test code needs to check whether
some variable is None to determine whether some hardware is present. On the other hand, if you need a hardware setting
just to run a script, it's not necessary to add a None default here.

For this file only, it should be safe for modules that require these settings to use
from kid_readout.settings import *
so the variables defined here must not conflict with anything else in the package. Ideally, use ALL_CAPS for constants.
Note that importing * from local_settings will cause its namespace to be imported here too, so keep it clean. Using e.g.
from kid_readout.roach.columbia import ROACH1_VALON, ROACH_IS_HETERODYNE
will import only the desired variables into the namespace, and not the module.
"""
import os
import socket


# TODO: move away from allowing HOSTNAME to determine code paths in analysis; use CRYOSTAT instead
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

try:
    from kid_readout.local_settings import *
except ImportError:
    pass
