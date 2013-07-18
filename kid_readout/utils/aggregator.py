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
        
    def gather(self, packet):
        
        # for each channel, push that onto the corresponding array.
        
        if self.packetCounter == packet.index[0]:
            for index in range(len(packet.data)):
                self.tmpbuffer[index][packet.index * 256:(packet.index + 1) * 256] = packet.data[index]
            # Takes into account channels
            self.packetCounter += 1
        else:
            if self.packetCounter > packet.index:
                pass;
                # Throws away late packets.
            if self.packetCounter < packet.index:
                for index in range(len(packet.data)):
                    self.tmpbuffer[index][(self.packetCounter * 256):((packet.index) * 256)] = float('NaN')
                # Fills discrepancy between counter and index with 'nan'
                for index in range(len(packet.data)):
                    self.tmpbuffer[index][packet.index * 256:(packet.index + 1) * 256] = packet.data[index]
                # Fills in correct index normally.
                self.packetCounter += 1
        
        if self.packetCounter == 16:
            self.create_data_products_from_gather(self.tmpbuffer, packet.clock)
            # pushes data to next function. Passes buffer and addr of packet index 16.
            
            self.packetCounter = 0
            # Resets packetCounter
            # For arbitrary buffer size, a mod function (packetCounter%16) could be used instead.
            
            # self.tmpbuffer = np.zeros((10, 256 * 16))
            # Reset tmpbuffer to zeros. If there is a way to easily initialize it to NaN's...
            # Resetting this created problems: the program would work correctly once and then the
            # tmpbuffer would no longer be written to. I took out the reset since all data will be overwritten
            # anyway.
    
    def gather_no_buff(self, packet):
        
        filldata = None
        if self.packetCounter == packet.index[0]:
            # send packet on
            self.packetCounter += 1
            self.lastpacket = packet
            self.create_data_products_experimental(packet)
        else:
            if self.packetCounter > packet.index:
                pass;
                # Throws away late packets.
            if self.packetCounter < packet.index:
                # Create filler packet and send it. All data should be NaN, copy other attributes from
                # previous packet.
                fillarray = np.empty(len(lastpacket.data[0]))
                fillarray[0:] = 'nan'
                
                filldata = range(len(lastpacket.data))
                for i in len(filldata):
                    filldata.append(fillarray)    
                newfiller = Filler(index, lastpacket.channel_id, lastpacket.clock, filldata)
                
                self.packetCounter += 1
                self.create_data_products_experimental(packet)
        
        if self.packetCounter == 16:
            
            self.packetCounter = 0
            # Resets packetCounter
            # For arbitrary buffer size, a mod function (packetCounter%16) could be used instead.
            
            # Could alternately just use a mod function. Maybe easier.
        
    def create_data_products_from_gather(self, passedArray, clockCount):
        pxx = (np.abs(np.fft.fft(passedArray[0])) ** 2)
        # Note that since we are currently testing with 1 channel, I only use the first channel of passed Array.
        # In the future there will need to be a pxx for each channel (for loop).
        # In reality, we will probably have more advanced data processing.
        pxx_product = dict(type='power spectrum', data=pxx, clock=clockCount)
        self.publish(pxx_product)
        
    def publishTest(self, data_product):
        print data_product['type']
        print data_product['data']
        print data_product['clock']
        # Used for debugging
    
        
    def create_data_products(self, chunk):
        # create higher level data products and distribute them to subscribers
        # this could be eventually moved to another thread or Pyro object etc to
        # improve CPU usage
        pxx = (np.abs(np.fft.fft(chunk['data'].reshape((-1, 1024)), axis=1)) ** 2).mean(0)
        pxx_product = dict(type='power spectrum', data=pxx)
        self.publish(pxx_product)
        
    def create_data_products_experimental(self, packet):
        pxx = (np.abs(np.fft.fft(packet.data[0])) ** 2)
        pxx_product = dict(type='power spectrum', data=pxx)
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
            
