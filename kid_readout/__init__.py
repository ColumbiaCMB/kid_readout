"""
Top level documentation test
"""

import logging
#  Follow recommendation in the Python logging HOWTO: make sure there is  a handler to avoid "No handlers found"
# error messages. Don't add any other handlers; that's up to the user application (i.e. data taking or analysis scripts)
logging.getLogger(__name__).addHandler(logging.NullHandler())


