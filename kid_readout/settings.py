"""
This is the global settings module for the package. It should contain values that are constant across systems and code
to determine values that can be discovered at runtime.

For this file only, it should be safe for modules that require these settings to use
from kid_readout.settings import *
so the variables defined here must not conflict with anything else in the package. Ideally, use ALL_CAPS for constants.

This module imports all local settings from local_settings.py, and these override the settings here. The local settings
file is not under version control. Every local setting should have a corresponding entry here, so that the test code can
intelligently check the value of the variable and skip the test if the current install doesn't support it for some
reason. An exception would be if the variable is really specific to a particular install. It's fine if the default
values given here aren't usable.
"""
import os
import socket


HOSTNAME = socket.gethostname()

# This is the directory into which data will be written. The default should always exist so that test code can use it.
BASE_DATA_DIR = os.path.join('/data', HOSTNAME)
if not os.path.exists(BASE_DATA_DIR):
    BASE_DATA_DIR = '/tmp'  # TODO: /data may not always exist, but /tmp should always be writable. Fine?

# The name of the cryostat. We could also
CRYOSTAT = None

# True if this system is connected to a roach. This should allow hardware tests to be skipped intelligently.
# Feel free to rename or add granularity, e.g. the exact model available. This could instead be something like the
# roach IP, with default None.
ROACH_AVAILABLE = False

# Import local settings
try:
    from .local_settings import *
except ImportError:
    pass
