import socket

class hittiteController():
    def __init__(self, addr=None, port=50000, terminator='\r'):
        if addr:
            self.address=(addr,port)
        else:
            self.address=('192.168.001.070',port)
            try:
                self.connect()
                self.disconnect()
            except socket.error:
                self.address=('192.168.0.070',port)

        self.terminator=terminator
        self.connect()
        
    def connect(self):
        s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        s.settimeout(0.5)
        s.connect(self.address)
        self.s = s
        return
    def disconnect(self):
        self.s.close()
        
    def send(self,msg):
        #self.connect()
        try:
            self.s.send(msg+self.terminator)
        except Exception,e:
            print e
            raise e
        finally:
            pass
        #    self.disconnect()
            
    def send_and_receive(self,msg):
        try:
            self.s.send(msg+self.terminator)
            response=self.s.recv(1024)
            return response
        except Exception,e:
            print e
            raise e
        finally:
            pass
            
    def on(self):
        self.send('OUTP ON')
    def off(self):
        self.send('OUTP OFF')
    def set_freq(self,freq):
        msg='FREQ %f'%(freq)
        self.send(msg)
    def set_power(self,power):
        msg='POW %f'%(power)
        self.send(msg)
            
    '''def on(self):
        self.connect()
        response = ''
        try:
            self.s.send('OUTP ON\r')
            response = self.s.recv(10000)
            return response
        except Exception,e:
            print e
            raise e
        finally:
            self.disconnect()'''
