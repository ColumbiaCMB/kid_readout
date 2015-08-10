"""
Agilent 33220A Function generator
"""
import socket
import time

class FunctionGenerator(object):
    def __init__(self,addr=('192.168.1.135', 5025)):
        self.addr = addr
        
    def set_load_ohms(self,ohms):
        self.send("OUTPUT:LOAD %d" % ohms)
        
    def set_dc_voltage(self,volts):
        self.send("APPLY:DC DEF, DEF, %f" % volts)
        
    def enable_output(self,on):
        if on:
            self.send("OUTPUT ON")
        else:
            self.send("OUTPUT OFF")

    def set_square_wave(self,freq,high_level,low_level=0,duty_cycle_percent=50.0):
        self.enable_output(False)
        self.send("FUNC SQUARE")
        self.send("FREQ %f" % freq)
        self.send("VOLT:HIGH %f" % high_level)
        self.send("VOLT:LOW %f" % low_level)
        self.send("FUNC:SQUARE:DCYCLE %f" % (duty_cycle_percent))
        time.sleep(1)
        print "waveform ready, remember to enable output."

    def set_pulse(self,period,width,high_level,low_level=0):
        if width >= period:
            raise ValueError("Width must be less than Period")
        self.enable_output(False)
        self.send("FUNC PULSE")
        self.send("FUNC:PULSE:HOLD WIDTH")
        time.sleep(2)
        self.send("PULSE:PERIOD %f" % period)
        time.sleep(0.3)
        self.send("PULSE:WIDTH %f" % width)
        time.sleep(0.3)
        self.send("VOLT:HIGH %f" % high_level)
        self.send("VOLT:LOW %f" % low_level)
        time.sleep(1)  # this sleep is required so that the output can be enabled immediately after, otherwise it
        # seems to ignore the enable request
        print "waveform ready, remember to enable output"
        
    def send_get(self,cmd,timeout=1):
        result = None
        try:
            s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
            s.connect(self.addr)
            s.settimeout(timeout)
            s.send(cmd+'\n')
            result = s.recv(1024)
        finally:
            s.close()
        return result
    
    def send(self,cmd):
        try:
            s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
            s.connect(self.addr)
            s.send(cmd+'\n')
            time.sleep(0.2)
        finally:
            s.close()
        
