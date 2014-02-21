"""
Agilent 33220A Function generator
"""
import socket

class FunctionGenerator(object):
    def __init__(self,addr=('192.168.1.135', 5025)):
        self.addr = addr
        
    def set_load_ohms(self,ohms):
        self.send("OUTPUT:LOAD %d" % ohms)
        
    def set_dc_voltage(self,volts):
        self.send("APPLY:DC DEF, DEF %f" % volts)
        
    def enable_output(self,on):
        if on:
            self.send("OUTPUT ON")
        else:
            self.send("OUTPUT OFF")
        
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
        finally:
            s.close()
        
