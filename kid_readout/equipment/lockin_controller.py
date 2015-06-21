import serial
import time
import numpy as np

sensitivities = ((np.array([1,2,5])[None,:])*((10.**np.arange(-9,1))[:,None])).flatten()[1:-2]
time_constants = ((np.array([1,3])[None,:])*((10.**np.arange(-5,5))[:,None])).flatten()


class lockinController():
### Initialization ###
    def __init__(self, serial_port='/dev/ttyUSB2', terminator='\n'):
        # For linux, theself.serial port will be something like the default address.
        # However, for windows it will be something like 'COM6'.
        self.address=serial_port
        self.baudrate=19200
        self.terminator=terminator
        self.ser=serial.Serial(self.address, baudrate=self.baudrate,timeout=1,rtscts=True)
        #self.ser.close()
        self.setup_rs232_output()
        
    
    def setup_rs232_output(self):
        #ser=serial.Serial(self.address, baudrate=self.baudrate)
        #self.ser.open()
        try:
           self.ser.write('OUTX 0'+self.terminator)
        except Exception as e:
            print e
            raise e
        finally:
            pass
#            self.ser.close()

### Basic send and receive methods ###

    def send(self,msg):
       #self.ser=serial.Serial(self.address, baudrate=self.baudrate)
#        self.ser.open()
        time.sleep(.1)
        try:
            self.ser.write(msg+self.terminator)
        except Exception as e:
            print e
            raise e
        finally:
            pass
#            self.ser.close()

    def send_and_receive(self,msg):
        #self.ser=serial.Serial(self.address, baudrate=self.baudrate, timeout=2)
#        self.ser.open()
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
#            self.ser.close()

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
