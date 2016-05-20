"""
This is the settings subpackage.

It imports default settings from _default.py as well as _local.py, if it exists, which contains site-specific settings.

The defaults are imported first, so the local settings will override the defaults. The local settings module is listed
in .gitignore and is thus not under version control so that deploys can have different settings.

If _local.py does not exist, this module will attempt to copy the file _autodetect.py -> _local.py, and if the copy
succeeds it will import * from this new _local.py file.

Ideally, it should be safe to use
from kid_readout.settings import *
so the variables defined here must not conflict with others in the package. If possible, use ALL_CAPS for constants and
prefix temporary variables with an underscore ('_') so that they are not carried into the namespace by import *. (The
module names in this subpackage also have underscore prefixes for this reason.)  Any module that is not built-in
should be imported with an underscore:
from kid_readout.subpackage import module as _module
Similarly, use
from kid_readout.subpackage.module import ONLY, NECESSARY, VARIABLES
to avoid importing the module name into the namespace.

Note that importing * from local.py will cause its namespace to be imported here too, so keep it clean.
"""
import os as _os
import shutil as _shutil
import socket as _socket
import logging as _logging
# Important: leading underscore helps ensure that logger in files won't be overridden by this logger
_logger = _logging.getLogger(__name__)

# Import default settings. Most of the values are None.
from kid_readout.settings._default import *

# Import local settings. If no local settings file exists and the hostname matches a key, copy the value to _local.py
# and import * from it.
_known_systems = {'detectors': '_detectors.py',
                  'readout': '_readout.py'}
_settings_dir = _os.path.dirname(_os.path.abspath(__file__))
_local_settings_filename = _os.path.join(_settings_dir, '_local.py')
if _os.path.exists(_local_settings_filename):
    from kid_readout.settings._local import *
else:  # Attempt to autodetect based on the hostname.
    _logger.warning("No local settings file found: {}".format(_local_settings_filename))
    _hostname = _socket.gethostname()
    if _hostname in _known_systems:
        _logger.warning("Trying to create a useful local settings file for {}.".format(_hostname))
        try:
            _shutil.copy(_os.path.join(_settings_dir, _known_systems[_hostname]), _local_settings_filename)
            _logger.info("Copied {} to _local.py and importing settings from it.".format(_known_systems[_hostname]))
            from kid_readout.settings._local import *
        except Exception as _exception:
            _logger.exception("Could not create _local.py: {}".format(_exception.message))
    else:
        _logger.info("Hostname {} is not a known system and no local settings were imported.".format(_hostname))
