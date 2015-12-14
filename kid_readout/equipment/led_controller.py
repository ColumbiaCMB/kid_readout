import serial
import time

class LedController(object):
    def __init__(self, port='/dev/ttyACM0',baudrate=9600):
        self.ser = serial.Serial(port=port,baudrate=baudrate,timeout=1)
    def sendget(self,msg,interchar_delay=0.1):
        if msg[-1] != '\n':
            msg = msg + '\n'
        for char in msg:
            self.ser.write(char)
            time.sleep(interchar_delay)
        time.sleep(1)
        resp = self.ser.read(self.ser.inWaiting())
        return resp
