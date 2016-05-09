"""
This module should contain up-to-date values for Columbia ROACH hardware.

Note that this file is under version control so it is shared by all deploys. If a piece of hardware has multiple common
settings, it's better to create multiple, descriptive names for these settings here instead of overwriting with the one
currently in use. The values in this file shouldn't have to change often.

On each system, local_settings.py should include only the settings for hardware that is actually connected. For example,
if a ROACH1 is connected and in baseband mode, local_settings.py would contain

from kid_readout import columbia
ROACH_IS_HETERODYNE = False
ROACH_HOST_IP = columbia.ROACH_HOST_IP
ROACH1_VALON = columbia.ROACH1_VALON

The package can use these values to automatically connect to the ROACH.
"""

# ROACH
ROACH_HOST_IP = '192.168.1.1'
ROACH1_VALON = '/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_AM01H05A-if00-port0'
ROACH1_IP = 'roach'
ROACH2_VALON = '/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_A101FK1K-if00-port0'
ROACH2_IP = 'r2kid'
MARK2_VALON = '/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_A101FK1H-if00-port0'
