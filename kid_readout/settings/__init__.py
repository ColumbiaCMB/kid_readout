"""
This is the settings subpackage.

It imports default settings from default.py as well as local.py, if it exists, which contains site-specific settings.

The defaults are imported first, so the local settings will override the defaults. The local settings module is listed
in .gitignore and is thus not under version control.



For this file only, it should be safe for modules that require these settings to use
from kid_readout.settings import *
so the variables defined here must not conflict with anything else in the package. Ideally, use ALL_CAPS for constants.
Note that importing * from local.py will cause its namespace to be imported here too, so keep it clean. Using e.g.
from kid_readout.roach.columbia import ROACH1_VALON, ROACH_IS_HETERODYNE
will import only the desired variables into the namespace, and not the module name.
"""
from kid_readout.settings.default import *

try:
    from kid_readout.settings.local import *
except ImportError:
    pass

