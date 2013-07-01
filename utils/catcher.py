import numpy as np
import time
import sys
import threading


class KatcpCatcher():
    def __init__(self,proc_func,bufname,roachip='roach'):
        self.bufname = bufname
        self.data_thread = None
        self.proc_func = proc_func
        from corr.katcp_wrapper import FpgaClient
        self.data_thread_r = FpgaClient(roachip,timeout=0.1)
        t1 = time.time()
        timeout = 10
        while not self.data_thread_r.is_connected():
            if (time.time()-t1) > timeout:
                raise Exception("Connection timeout to roach")
            time.sleep(0.1)
            
        self.last_addr=0


    def start_data_thread(self):
        if self.data_thread:
            self.quit_data_thread = True
            self.data_thread.join(1.0)
            self.data_thread = None
        self.quit_data_thread = False
        self.data_thread = threading.Thread(target=self._cont_read_data,args=())
        self.data_thread.daemon = True
        self.data_thread.start()
        
    def _proc_raw_data(self,data,addr):
        if addr - self.last_addr > 8192:
            print "skipped:", addr,self.last_addr,(addr-self.last_addr)
        self.last_addr = addr 
        data = np.fromstring(data,dtype='>i2').astype('float32').view('complex64')
        self.pxx = (np.abs(np.fft.fft(data.reshape((-1,1024)),axis=1))**2).mean(0)
        
    def _cont_read_data(self):
        """
        Low level data reading loop. Reads data continuously and passes it to self.proc_func
        """
        regname = '%s_addr' % self.bufname
        brama = '%s_a' % self.bufname
        bramb = '%s_b' % self.bufname
        r = self.data_thread_r
        a = r.read_uint(regname) & 0x1000
        addr = r.read_uint(regname) 
        b = addr & 0x1000
        while a == b:
            addr = r.read_uint(regname)
            b = addr & 0x1000
        data = []
        addrs = []
        tic = time.time()
        idle = 0
        while not self.quit_data_thread:
            a = b
            if a:
                bram = brama
            else:
                bram = bramb
            data = r.read(bram,4*2**12)
            self.proc_func(data,addr)
            
            addr = r.read_uint(regname)
            b = addr & 0x1000
            while (a == b) and not self.quit_data_thread:
                try:
                    addr = r.read_uint(regname)
                    b = addr & 0x1000
                    idle += 1
                except Exception, e:
                    print e
                time.sleep(0.1)
        else:
            print "exiting data thread"
            
