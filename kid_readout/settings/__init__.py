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
prefix all variables with an underscore ('_') so that they are not carried into the namespace by import *. (All of the
module names in this subpackage also have underscore prefixes for this reason.) Import module names similarly:
from kid_readout.subpackage import module as _module
and use
from kid_readout.subpackage.module import ONLY, NECESSARY, VARIABLES
to avoid importing the module name into the namespace.

Note that importing * from local.py will cause its namespace to be imported here too, so keep it clean.
"""
import os as _os
import shutil as _shutil
import logging as _logging
# Important: leading underscore helps ensure that logger in files won't be overridden by this logger
_logger = _logging.getLogger(__name__)

from kid_readout.settings._default import *

_settings_dir = _os.path.dirname(_os.path.abspath(__file__))
_local_settings_filename = _os.path.join(_settings_dir, '_local.py')
if _os.path.exists(_local_settings_filename):
    from kid_readout.settings._local import *
else:
    _logger.warning("No kid_readout/settings/_local.py file found. Trying to create a useful default.")
    try:
        _shutil.copy(_os.path.join(_settings_dir, '_autodetect.py'), _local_settings_filename)
        _logger.info("Successfully created _local.py file.")
    except Exception as _exception:
        _logger.exception("Could not create _local.py: {}".format(_exception.message))
    from kid_readout.settings._local import *