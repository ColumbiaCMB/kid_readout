import time
import threading
import logging 
import serial 
import io
import sim900
import sys

if __name__ == "__main__":

    #this is a bad file for recording the diode temps and voltages
    #eventually it will be merged with recording the resistance bridges
    #and actually use the sim900 file functions

    #create an instance of the sim900 commands
    sim = sim900.sim900(port='/dev/ttyUSB2')


    #main function to records temps
    try:
        timestr = time.strftime("%Y%m%d-%H%M%S")
        filename = "/home/data/SRS/%s.txt" % timestr
        f = open(filename, 'w+')
        f.write("time, diode ch1 temp, dio ch 2 temp, dio 3 temp, dio 4 temp, dio 1 volts, dio 2 volts, dio 3 volts, dio 4 volts, rox 1 temp, rox 1 res, rox 2 temp, rox 2 res, rox 3 res, rox 3 temp\n")

        while 1:

            #get diode info
            sim.connect_sim922()
            dio_temps = sim.get_sim922_temp()
            dio_temps = dio_temps.rstrip()
            time.sleep(1)
            dio_volts = sim.get_sim922_volts()
            dio_volts = dio_volts.rstrip()
            sim.close_sim922()
            print "diode"

            time.sleep(1)

            #get rox1 info
            sim.connect_sim921_1()
            rox1_res = sim.get_resistance()
            rox1_temp = sim.get_temp()
            sim.close_sim921_1()

            print "rox1"

            time.sleep(1)

            sim.connect_sim921()
            rox2_res = sim.get_resistance()
            rox2_temp = sim.get_temp()
            sim.close_sim921()


            #get rox3 info
            sim.connect_sim921_6()
            rox3_res = sim.get_resistance()
            rox3_temp = sim.get_temp()
            sim.close_sim921_6()

            print "rox2"

            time.sleep(1)

            #write it all to file
            current_time = time.strftime("%Y%m%d-%H%M%S")
            f.write("%s, %s, %s, %s, %s, %s, %s, %s, %s\n" % (current_time, dio_temps, dio_volts, rox1_temp, rox1_res, rox2_temp, rox2_res, rox3_temp, rox3_res))
            f.flush()

    except KeyboardInterrupt:
        f.close()
        print "done writing"
        sim.close_sim922()
        sim.close_sim900()
        print "ports closed"
