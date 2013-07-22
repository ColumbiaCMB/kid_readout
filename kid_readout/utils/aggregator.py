import numpy as np
import time
import sys
import single_pixel
import threading
import Pyro4
import collections as col


class Aggregator():
    def __init__(self, parent, writer):
        self.parent = parent
        self.writer = writer
        self.subscriptions = {'power spectrum': []}
        self.subscribers = {}  # maps uri to subscriber
        self.last_addr = 0
        
        self.packetCounter = 0
        # Used for keeping track of the packet order.
        self.tmpbuffer = np.zeros((10, 256 * 16)).view('complex64')
        # Used for aggregating packets for ordering.
        Filler = col.namedtuple('Packet', ['index', 'channel_id', 'clock', 'data'])
        self.lastpacket = None
        
        
    def subscribe_uri(self, uri, data_products):
        if self.subscribers.has_key(uri):
            print "already subscribed!", uri
            return
        subscriber = Pyro4.Proxy(uri)
        self.subscribers[uri] = subscriber
        for data_product in data_products:
            self.subscriptions[data_product].append(subscriber)

    def proc_raw_data(self, data, addr):
        if addr - self.last_addr > 8192:
            print "skipped:", addr, self.last_addr, (addr - self.last_addr)
        self.last_addr = addr 
        data = np.fromstring(data, dtype='>i2').astype('float32').view('complex64')
        # create chunk representation of this data
        chunk = dict(data=data, time=time.time())
        self.writer.write_data_chunk(chunk)
        self.create_data_products(chunk)
        
    def publish_test(self, data_product):
        print data_product['type']
        print data_product['data']
        print data_product['clock']
        # Used for debugging
    
        
    def create_data_products(self, chunk):
        # create higher level data products and distribute them to subscribers
        # this could be eventually moved to another thread or Pyro object etc to
        # improve CPU usage
        pxx = (np.abs(np.fft.fft(chunk['data'].reshape((-1, 1024))  , axis=1)) ** 2).mean(0)
        pxx_product = dict(type='power spectrum', data=pxx)
        self.publish(pxx_product)
        
    def create_data_products_short(self, packet):
        pxx = (np.abs(np.fft.fft(packet.data[0])) ** 2)
        pxx_product = dict(type='power spectrum', data=pxx, clock=packet.clock)
        self.publish(pxx_product)
        
    def publish(self, data_product):
        # pass the data products on to the appropriate subscribers
        subscription_list = self.subscriptions[data_product['type']]  # list of subscribers interested in this data product
        print "publishing pxx to", subscription_list
        for subscriber in subscription_list:
            subscriber.handle(data_product)
        
    def _cont_read_data(self):
        """
        Low level data reading loop. Reads data continuously and passes it to self._proc_raw_data
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
            data = r.read(bram, 4 * 2 ** 12)
            self._proc_raw_data(data, addr)
            
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
            
