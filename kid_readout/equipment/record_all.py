import os
import time
from kid_readout.equipment import sim

def main():

    # Time between temperature requests, in seconds.
    delay = 3

    basepath = '/dev/serial/by-id'
    serial_id = 'usb-FTDI_USB_to_Serial_Cable_FTGQM0GY-if00-port0'
    serial_port = os.path.realpath(os.path.join(basepath, serial_id))
    sim900 = sim.SIM900(serial_port)
    print("Connected to {}".format(sim900.identification))
    print("Port: Connected device:")
    for port, device in sim900.ports.items():
        print("{}     {}".format(port, device))

    # Set up the SIMs
    ruox3628 = sim900.ports['4']
    ruox3882 = sim900.ports['6']
    diodes = sim900.ports['8']

    ruox3628.reset()
    ruox3628.excitation = 2 # 30 uV
    ruox3628.excitation_mode = 'VOLTAGE'
    ruox3628.time_constant = 2 # 3 s
    ruox3628.autorange_gain()
    ruox3628.display_temperature = True
    ruox3628.curve_number = 1
    print("Port 4 curve: {}".format(ruox3628.curve_info(ruox3628.curve_number)[1]))

    ruox3882.reset()
    ruox3882.excitation = 2 # 30 uV
    ruox3882.excitation_mode = 'VOLTAGE'
    ruox3882.time_constant = 2
    ruox3882.autorange_gain()
    ruox3882.display_temperature = True
    ruox3882.curve_number = 1
    print("Port 6 curve: {}".format(ruox3882.curve_info(ruox3882.curve_number)[1]))

    diodes.reset()
    diodes.set_curve_type(1, 'USER')
    print("Port 8 channel 1 curve: {}".format(diodes.curve_info(1)[1]))
    diodes.set_curve_type(2, 'USER')
    print("Port 8 channel 2 curve: {}".format(diodes.curve_info(2)[1]))

    try: 
        header = "time, diode ch1 temp, dio ch 2 temp, dio 3 temp, dio 4 temp, dio 1 volts, dio 2 volts, dio 3 volts, dio 4 volts, rox 1 temp, rox 1 res, rox 2 temp, rox 2 res, rox 3 temp, rox 3 res"
        timestr = time.strftime("%Y%m%d-%H%M%S")
        filename = "/home/data/SRS/%s.txt" %timestr
        print("Writing to {}".format(filename))
        f = open(filename, 'w+')
        f.write(header + '\n')
        f.flush()
        print(header)

        while True:
            dio1_volt = diodes.voltage(1)
            dio1_temp = diodes.temperature(1)
            dio2_volt = diodes.voltage(2)
            dio2_temp = diodes.temperature(2)
            # Update this if we attach diodes to channels 3 and 4
            dio3_volt = dio3_temp = dio4_volt = dio4_temp = 0
            rox1_res = ruox3628.resistance
            rox1_temp = ruox3628.temperature
            rox2_res = ruox3882.resistance
            rox2_temp = ruox3882.temperature
            # Update this if we attach a new SIM921
            rox3_res = rox3_temp = 0
    
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
