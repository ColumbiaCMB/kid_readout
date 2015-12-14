import os
import time
from kid_readout.equipment import sim

def main():

    # Time between temperature requests, in seconds.
    delay = 3

    basepath = '/dev/serial/by-id'
    serial_id = 'usb-FTDI_USB_to_Serial_Cable_FTGQM0GY-if00-port0'
    serial_port = os.path.realpath(os.path.join(basepath, serial_id))
    sim900 = sim.SIM900(serial_port, baudrate=115200)
    print("Connected to {}".format(sim900.identification))
    print("Port: Connected device:")
    for port, device in sim900.ports.items():
        print("{}     {}".format(port, device.__class__.__name__))

    # Set up the SIMs
    sim1 = sim900.ports['1']
    ruox3628 = sim900.ports['4']
    ruox3882 = sim900.ports['6']
    diodes = sim900.ports['8']

    # Default RuOx settings for measuring millikelvin temperatures
    excitation = 2
    excitation_mode = 'VOLTAGE'
    range = 6  # 20 kOhm
    time_constant = 2  # 3 s
    display_temperature = True

    sim1.reset()
    sim1.excitation = excitation
    sim1.excitation_mode = excitation_mode
    sim1.range = range
    sim1.time_constant = time_constant


    ruox3628.reset()
    ruox3628.excitation = excitation
    ruox3628.excitation_mode = excitation_mode
    ruox3628.range = range
    ruox3628.time_constant = time_constant
    ruox3628.display_temperature = display_temperature
    ruox3628.curve_number = 1
    #ruox3628.autorange_gain()
    print("Port 4 curve: {}".format(ruox3628.curve_info(ruox3628.curve_number)[1]))

    ruox3882.reset()
    ruox3882.excitation = excitation
    ruox3882.excitation_mode = excitation_mode
    ruox3882.range = 5  # 2 kOhm because the Eccosorb is at least 1 K.
    ruox3882.time_constant = time_constant
    ruox3882.display_temperature = display_temperature
    ruox3882.curve_number = 1
    # Set up the analog output
    ruox3882.temperature_setpoint = 0
    ruox3882.analog_output_temperature = True
    #ruox3882.analog_output_manual_value = 10
    #ruox3882.analog_output_manual_mode = True
    #ruox3882.autorange_gain()
    #ruox3882.analog_output_manual_mode = False
    print("Port 6 curve: {}".format(ruox3882.curve_info(ruox3882.curve_number)[1]))

    diodes.reset()
    diodes.set_curve_type(1, 'USER')
    print("Port 8 channel 1 curve: {}".format(diodes.curve_info(1)[1]))

    try: 
        header = "time, diode ch1 temp, dio ch 2 temp, dio 3 temp, dio 4 temp, dio 1 volts, dio 2 volts, dio 3 volts, dio 4 volts, rox 1 temp, rox 1 res, rox 2 temp, rox 2 res, rox 3 temp, rox 3 res"
        timestr = time.strftime("%Y%m%d-%H%M%S")
        filename = "/data/readout/SRS/%s.txt" %timestr
        print("Writing to {}".format(filename))
        f = open(filename, 'w+')
        f.write(header + '\n')
        f.flush()
        print(header)

        while True:
            dio1_volt = diodes.voltage(1)
            dio1_temp = diodes.temperature(1)
            dio2_volt = dio2_temp = dio3_volt = dio3_temp = dio4_volt = dio4_temp = 0
            rox1_res = ruox3628.resistance
            rox1_temp = ruox3628.temperature
            rox2_res = ruox3882.resistance
            rox2_temp = ruox3882.temperature
            rox3_res = sim1.resistance
            rox3_temp = sim1.temperature

            current_time = time.strftime("%Y%m%d-%H%M%S")
            all_values = ", ".join([str(n) for n in
                                    (current_time,
                                     dio1_temp, dio2_temp, dio3_temp, dio4_temp,
                                     dio1_volt, dio2_volt, dio3_volt, dio4_volt,
                                     rox1_temp, rox1_res, rox2_temp, rox2_res, rox3_temp, rox3_res)])
            f.write(all_values+'\n')
            f.flush()
            print(all_values)
            time.sleep(delay)

    except KeyboardInterrupt:
        f.close()
        sim900.disconnect()
        sim900.serial.close()

if __name__ == "__main__":
    main()
