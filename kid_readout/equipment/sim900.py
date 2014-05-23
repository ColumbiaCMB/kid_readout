import time
import threading
import logging 
import serial 
import io
import sys 

class sim900():
    def __init__(self):
        #open port, connect to sim900
        #self.ser = serial.Serial(0, baudrate = 9600, timeout = 2)
        self.ser = serial.Serial(port="/dev/ttyUSB1", baudrate=9600, timeout=2)
        self.setup_sim900()
    def setup_sim900(self):
        #flush all port butters
        self.ser.write("FLSH\n")
        #reset module interfaces
        self.ser.write("SRST\n")
        self.ser.write("*IDN?\n")
        self.msg = self.ser.readline()
        sys.stdout.flush()
        print self.msg
    
    def close_sim900(self):
        self.close_sim921() #to be safe
        self.ser.close()
        self.ser.isOpen()


    def connect_sim921(self):
        #connect to sim921
        self.ser.write("CONN 4, 'xxx'\n")
        self.ser.write("*IDN?\n")
        self.msg = self.ser.readline()
        print self.msg
        sys.stdout.flush()  

    def close_sim921_1(self):
        self.ser.write("xxx\n")

    def connect_sim921_1(self):
        #connect to sim921
        self.ser.write("CONN 1, 'xxx'\n")
        self.ser.write("*IDN?\n")
        self.msg = self.ser.readline()
        print self.msg
        sys.stdout.flush()

    def close_sim921_6(self):
        self.ser.write("xxx\n")
    def connect_sim921_6(self):
        #connect to sim921
        self.ser.write("CONN 6, 'xxx'\n")
        self.ser.write("*IDN?\n")
        self.msg = self.ser.readline()
        print self.msg
        sys.stdout.flush()

    def close_sim921(self):
        self.ser.write("xxx\n")    

    def setup_sim921(self):
        self.ser.write("CURV 2\n")#set calibration curve to use
        self.ser.write("DTEM 1\n")#set display in Kelvin
        self.ser.write("AGAI 1\n")#let the gain autorange
        self.ser.write("ADIS 1\n")#let the display autorange

    def set_sim921_excitation(self):
        self.ser.write("FREQ 10\n") #set frequency, at 10hz -CHECK
        
        #we want .3microA going across the RuOx600- CHECK
        self.ser.write("RANG 6\n") #set range to 20 kohms (5) 
        self.ser.write("MODE 2\n") #set excitation mode to voltage  biased
        self.ser.write("EXON 1\n") #set exctiation mode on
        self.ser.write("EXCI 2\n") #set excitiation to 30 microVolts (4)
        
        
    def load_calibration_curve(self):
        #this program loads a calibration curve into the 1st slot
        self.ser.write("CINI 1, 0, 'roxcal'\n") #initialize calibration
        self.curve_num = 'CAPT 1'
        self.ender = '\n'
        #get calibration file
        self.curve_file = open('3628data_reversed.txt', 'r')
        self.curve = self.curve_file.readlines()
        self.curve_file.close()
        
        for i in xrange(len(self.curve)):
            self.p = self.curve[i].split()
            self.resistance, self.temp = self.p[1], self.p[0]
              
            self.cal_message = [self.curve_num, str(self.p[1]), str(self.p[0])]
            self.cal_message = (",".join(self.cal_message) + self.ender) #format it
            
            self.ser.write('%s' % self.cal_message) #feed in data points (cure, res, temp)
            time.sleep(1)
            
            print self.cal_message
             
    
    def load_calibration_curve_2(self):
        #this program loads a calibration curve into the 1st slot
        self.ser.write("CINI 1, 0, 'roxstan'\n") #initialize calibration
        self.curve_num = 'CAPT 1'
        self.ender = '\n'
        
        #get calibration file
        self.curve_file = open('RO600.txt', 'r')
        self.curve = self.curve_file.readlines()
        self.curve_file.close()
        
        for i in xrange(len(self.curve)):
            self.p = self.curve[i].split()
            self.resistance, self.temp = self.p[0], self.p[1]
              
            self.cal_message = [self.curve_num, str(self.p[0]), str(self.p[1])]
            print self.cal_message
            self.cal_message = (",".join(self.cal_message) + '\n') #format it
            
            self.ser.write('%s' % self.cal_message) #feed in data points (cure, res, temp)
            time.sleep(1)
            
            print self.cal_message
            print self.ender   
    

    def load_calibration_curve_diode(self):
        #this program loads a calibration curve into the 1st slot
        self.ser.write("CINI 1, 0, 'diode1'\n") #initialize calibration

        self.curve_num = 'CAPT 1'
        self.ender = '\n'

        #get calibration file
        self.curve_file = open('D6042351.txt', 'r')
        self.curve = self.curve_file.readlines()
        self.curve_file.close()

        for i in xrange(len(self.curve)):
            self.p = self.curve[i].strip().split()
            self.temp, self.volt = self.p[0], self.p[1]
            
            self.cal_message = [self.curve_num, str(self.p[1]), str(self.p[0])]

            self.cal_message = (",".join(self.cal_message) + '\n') #format it
            
            print self.cal_message

            self.ser.write('%s' % self.cal_message) #feed in data points (cure, res, temp)
            time.sleep(2)

            print self.cal_message
            print self.ender   
        
        self.ser.write("CURV 1,1\n")
        


    def check_cal(self, entry):
        self.ender = '\n'
        self.cal_message = ['CAPT? 2', str(entry)]

        self.cal_message = (",".join(self.cal_message) + self.ender)
        
        self.ser.write('%s' % self.cal_message)
        self.msg = self.ser.readline()
        print self.msg

    def get_resistance(self):
        self.ser.write("RVAL?\n")
        self.msg = self.ser.readline()
     #sys.stdout.flush()    
        self.ser.flush()
        return float(self.msg)
    
    def get_temp(self):
        self.ser.write("TVAL?\n")
        self.msg = self.ser.readline()
     #sys.stdout.flush()
        self.ser.flush()
        return float(self.msg)

    def get_sim922_temp(self):
        self.ser.write("TVAL? 0\n")
        self.msg = self.ser.readline()
        self.ser.flush()
        return str(self.msg)

    def get_sim922_volts(self):
        self.ser.write("VOLT? 0\n")
        self.msg = self.ser.readline()    
        self.ser.flush()
        return str(self.msg)

    def connect_sim925(self):
        #connect to sim925, multiplexer
        self.ser.write("CONN 3, 'xyz'\n")
        self.ser.write("*IDN?\n")
        self.msg = self.ser.readline()
        #print self.msg
        sys.stdout.flush()

    def close_sim925(self):
        self.ser.write("xyz\n")

    def set_channel_1(self):
        self.ser.write("CHAN 1\n")

    def set_channel_2(self):
        self.ser.write("CHAN 2\n")

    def connect_sim922(self):
        #connect to sim922, diode 
        self.ser.write("CONN 8, 'zzz'\n")
        self.ser.write("*IDN?\n")
        self.msg = self.ser.readline()
        print self.msg
        sys.stdout.flush()

    def close_sim922(self):
        self.ser.write("zzz\n")
        print self.msg
        sys.stdout.flush()


    def sendget(self, command):
        if command[-1] != '\n':
            command = command + '\n'
        self.ser.write(command)
        time.sleep(.2)

        self.msg = self.ser.readline()
        print self.msg

#sndt, port number + message 
