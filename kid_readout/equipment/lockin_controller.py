import serial
import time
import numpy as np

sensitivities = ((np.array([1,2,5])[None,:])*((10.**np.arange(-9,1))[:,None])).flatten()[1:-2]
time_constants = ((np.array([1,3])[None,:])*((10.**np.arange(-5,5))[:,None])).flatten()


class lockinController():
    def __init__(self, serial_port='/dev/ttyUSB2', terminator='\n'):
        # For linux, the serial port will be something like the default address.
        # However, for windows it will be something like 'COM6'.
        self.address=serial_port
        self.baudrate=19200
        self.terminator=terminator
        self.ser=serial.Serial(self.address, baudrate=self.baudrate,timeout=1,rtscts=True)
        self.time_constant_value = -1
        self.sensitivity_value = -1
        self.setup_rs232_output()


    def setup_rs232_output(self):
        try:
           self.ser.write('OUTX 0'+self.terminator)
        except Exception as e:
            print e
            raise e
        finally:
            pass

    def send(self,msg):
        time.sleep(.1)
        try:
            self.ser.write(msg+self.terminator)
        except Exception as e:
            print e
            raise e
        finally:
            pass

    def send_and_receive(self,msg):
        time.sleep(.1)
        # This delay is necessary... for some reason. The system behaves very poorly otherwise.
        # Perhaps theself.serial port takes some time to initialize?
        try:
            self.ser.write(msg+self.terminator)
            return self.read_until_terminator()
        except Exception as e:
            print e
            raise e
        finally:
            pass

    def get_sensitivity(self):
        response = int(self.send_and_receive('SENS?'))
        self.sensitivity_value = response
        return response, sensitivities[response]

    def set_sensitivity(self,sensitivity_volts):
        idx = sensitivities.searchsorted(sensitivity_volts)
        if idx >= len(sensitivities):
            idx = len(sensitivities)-1
        self.send("SENS %d" % idx)
        self.sensitivity_value = idx

    def get_time_constant(self):
        response = int(self.send_and_receive('OFLT?'))
        self.time_constant_value = response
        return response, time_constants[response]

    def auto_range_measure(self,debug=False):
        if self.sensitivity_value < 0:
            idx,sens = self.get_sensitivity()
            if debug:
                print "initialized sensitivity:", idx,sens
        x,y,r,theta = self.get_data()
        sens = sensitivities[self.sensitivity_value]
        if debug:
            print "initial sensitivity: %g, initial measurement %g" % (sens,r)
        while r > sens/2.0:
            self.set_sensitivity(r*10)
            sens = sensitivities[self.sensitivity_value]
            if debug:
                print "setting sensitivity to %g. new value %g" % (r*10,sens)
            time.sleep(0.1)
            x,y,r,theta = self.get_data()
        if sens > 100e-9:
            while r < sens/1000.0:
                self.set_sensitivity(r/100.0)
                sens = sensitivities[self.sensitivity_value]
                if debug:
                    print "setting sensitivity to %g. new value %g" % (r/1000,sens)
                time.sleep(0.1)
                x,y,r,theta = self.get_data()
        return r, sens

    def read_until_terminator(self):
        message=''
        new_char=None
        while new_char!='\r':
            new_char= self.ser.read(1)
            if new_char=='':
                # This meansself.ser has timed out. We don't want an unending loop if the terminator has somehow been lost.
                print 'Serial port timed out while reading.'
                break
            message+=new_char
        return message

    ### Stub methods ###

    def get_data(self):
        data_string = self.get_snap()
        data_string=data_string.strip('\r')
        data_list=data_string.split(',')
        try:
            x = float(data_list[0])
            y = float(data_list[1])
            r = float(data_list[2])
            theta = float(data_list[3])
            return x,y,r,theta
        except Exception:
            return 0,0,0,0
    def get_snap(self):
        return self.send_and_receive('SNAP? 1,2,3,4')

    def get_idn(self):
        return self.send_and_receive('*IDN?')
