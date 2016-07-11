"""
This module should contain up-to-date values for Columbia ROACH hardware.

Note that this file is under version control so it is shared by all deploys. If a piece of hardware has multiple common
settings, it's better to create multiple, descriptive names for these settings here instead of overwriting with the one
currently in use. The values in this file shouldn't have to change often.

On each system, local.py should contain or import only the settings for hardware that is actually connected. For
example, if a ROACH1 is connected and in baseband mode, local.py could contain

from kid_readout.settings.roach import ROACH1_IP, ROACH1_VALON, ROACH1_HOST_IP
"""

# ROACH1
ROACH1_IP = 'roach'
ROACH1_VALON = '/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_AM01H05A-if00-port0'
ROACH1_HOST_IP = '192.168.1.1'

# ROACH2
ROACH2_IP = 'r2kid'
ROACH2_VALON = '/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_A101FK1K-if00-port0'
ROACH2_HOST_IP = '192.168.1.1'
ROACH2_GBE_HOST_IP = '10.0.0.1'

MARK2_VALON = '/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_A101FK1H-if00-port0'
