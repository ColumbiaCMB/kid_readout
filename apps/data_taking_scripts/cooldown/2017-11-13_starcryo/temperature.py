from __future__ import print_function
import os
import time

from kid_readout.settings import (SRS_TEMPERATURE_SERIAL_PORT,
                                  SRS_TEMPERATURE_BAUD_RATE,
                                  TEMPERATURE_LOG_DIR)
from equipment.srs import sim


def write_and_print(filename, values, separator):
    with open(filename, 'a+') as f:  # Open in append mode with reading allowed
        print(*values, sep=separator, file=f)
        print(*values, sep=separator)


def main():
    separator = ', '
    # Time between temperature requests, in seconds.
    delay = 3
    # Time to display curve name, in seconds
    curve_display = 0.5

    sim900 = sim.SIM900(serial_port=SRS_TEMPERATURE_SERIAL_PORT,
                        baudrate=SRS_TEMPERATURE_BAUD_RATE)
    print("Connected to {}".format(sim900.identification))
    print("Port: Connected device:")
    for port, device in sim900.ports.items():
        print("{}     {}".format(port, device))

    # Set up the SIMs
    ruox4550 = sim900.ports['6']
    diodes = sim900.ports['8']

    # Default RuOx settings for measuring millikelvin temperatures
    excitation = 2
    excitation_mode = 'VOLTAGE'
    resistance_range = 6  # 20 kOhm
    time_constant = 2  # 3 s
    display_temperature = True
    
    for sim921 in (ruox4550,):
        sim921.reset()
        sim921.excitation = excitation
        sim921.excitation_mode = excitation_mode
        sim921.range = resistance_range
        sim921.time_constant = time_constant
        sim921.display_temperature = display_temperature

    diodes.reset()
    diodes.set_curve_type(1, 'USER')
    print("Port 8 channel 1 curve: {}".format(diodes.curve_info(1)[1]))

    try: 
        header = ("date_and_time", "unix_time",
                  "eccosorb_diode_voltage", "eccosorb_diode_temperature",
                  "stepper_diode_voltage", "stepper_diode_temperature",
                  "package_ruox4550_resistance", "package_ruox4550_temperature")
        filename = os.path.join(TEMPERATURE_LOG_DIR, "{}.txt".format(time.strftime("%Y%m%d-%H%M%S")))
        if os.path.exists(filename):
            raise ValueError('File already exists: {}'.format(filename))
        else:
            print("Writing to {}".format(filename))
        write_and_print(filename, header, separator)
        while True:
            unix_time = time.time()
            values = (time.strftime("%Y%m%d-%H%M%S", time.localtime(unix_time)), unix_time,
                      diodes.voltage(1), diodes.temperature(1),
                      diodes.voltage(2), diodes.temperature(2),
                      ruox4550.resistance, ruox4550.temperature)
            write_and_print(filename, values, separator)
            for s in (ruox4550,):
                s.display = 0  # Display the curve name
            time.sleep(curve_display)
            for s in (ruox4550,):
                s.display = 8  # Display the temperature or resistance
            time.sleep(delay - curve_display)
    finally:
        sim900.disconnect()
        sim900.serial.close()


if __name__ == "__main__":
    main()
