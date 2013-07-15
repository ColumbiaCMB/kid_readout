import numpy as np
import time
import sys
import threading
import socket
import collections as col



class PacketCatcher():
    def __init__(self, publish_func, bufname, roachip='roach'):
        self.bufname = bufname
        self.data_thread = None
        self.publish_func = publish_func
        self.last_addr = 0
        
    def start_data_thread(self):
        if self.data_thread:
            self.quit_data_thread = True
            self.data_thread.join(1.0)
            self.data_thread = None
        self.quit_data_thread = False
        self.data_thread = threading.Thread(target=self._cont_read_data, args=("192.168.1.1", 12345))
        # Using the port and IP startup_server runs on for now.
        self.data_thread.daemon = True
        self.data_thread.start()
        
    def decode(self, pkt, chan):
    
        index = np.fromstring(pkt[:2], dtype='>i2')
        channel_id = np.fromstring(pkt[2:4], dtype='>i2')
        addr = np.fromstring(pkt[4:8], dtype='>u4')
        data_list = np.fromstring(pkt[8:], dtype='>i2').astype('float').view('complex')
        '''Read data as an array of 2byte integers --> convert to float --> view as
        complex pairs (real and imaginary)'''
        
        data = range(chan)
        for i in range(chan):
            data[i] = data_list[i::chan]
        
        Packet = col.namedtuple('Packet', ['index', 'channel_id', 'addr', 'data'])
        myPacket = Packet(index, channel_id, addr, data_list)
        xmyPacket = Packet(index, channel_id, addr, data)
    
        return myPacket
    
    def _cont_read_data(self, UDP_IP, UDP_PORT):
        """
        Reads data from socket as a chunk. Passes it to a publishing function passed
        to UDPCatcher (intended to be aggregator.create_data_products).
        """
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((UDP_IP, UDP_PORT))
        
        while not self.quit_data_thread:
            raw_data = sock.recv(10000)
            packet = self.decode(raw_data, 5)
            self.publish_func(packet)
            # This is where data will be pushed to aggregator, and in turn to subscribers.
            # Intended aggregator function is create_data_products_udp


class Packet:
    def __init__(self, raw):
        self.decode(raw)
    def decode(self, pkt):
        self.index = np.fromstring(pkt[:2], dtype='>i2')
        self.channel_id = np.fromstring(pkt[2:4], dtype='>i2')
        self.addr = np.fromstring(pkt[4:8], dtype='>u4')
        self.data = np.fromstring(pkt[8:], dtype='>i2').astype('float').view('complex')
        '''Read data as an array of 2byte integers --> convert to float --> view as
        complex pairs (real and imaginary)'''
        
class Chunk:
    def __init__(self, socket, channel_num):
        self.get_data(self.gather(socket), channel_num)
    def get_data(self, pktlist, channel_num):
        allsize = channel_num * (((16 * 256) / channel_num) + 1)
        # This messy arithmatic expression just ensures all is long enough.
        # We will have 16 packets each of 256 numbers. The rest makes sure the array is divisible by the number
        # of channels and won't overflow.  
        all = np.zeros(allsize, dtype='complex') 
        self.order = []
        i = 0
        while(i < 16):
            all[(i * 256):((i + 1) * 256)] = pktlist[i].data
            i += 1
        all = all.reshape((allsize / channel_num), channel_num)
        self.data = all
        self.channel_id = pktlist[0].channel_id
        self.addr = pktlist[0].addr
    def gather(self, socket):
        tmplist = []
        olist = []
        i = 0
        while(i < 16):
            raw_input = socket.recv(10000)
            pkt = Packet(raw_input)
            olist.append(pkt.index[0])
            # Error checking. Should test more rigorously.
            if(pkt.index[0] != i):
                print "Packets arrived out of order."
                if(pkt.index[0] > i):
                    print "Packet skipped. Updated from current packet"
                    filler = Packet(("\0"*1032))
                    filler.data = np.zeros(256)
                    for skip in range(pkt.index[0], i):
                        tmplist.append(filler)
                    tmplist.append(pkt)
                if(pkt.index[0] < i):
                    print "Late packet: discarded"
                    i -= 1
            else:
                tmplist.append(pkt)
            i += 1
        # print olist
        # Useful for testing order of packets.
        return tmplist


class UDPCatcher():
    def __init__(self, publish_func, bufname, roachip='roach'):
        self.bufname = bufname
        self.data_thread = None
        self.publish_func = publish_func
        self.last_addr = 0
        
    def start_data_thread(self):
        if self.data_thread:
            self.quit_data_thread = True
            self.data_thread.join(1.0)
            self.data_thread = None
        self.quit_data_thread = False
        self.data_thread = threading.Thread(target=self._cont_read_data, args=("192.168.1.1", 12345))
        # Using the port and IP startup_server runs on for now.
        self.data_thread.daemon = True
        self.data_thread.start()
    
    def _cont_read_data(self, UDP_IP, UDP_PORT):
        """
        Reads data from socket as a chunk. Passes it to a publishing function passed
        to UDPCatcher (intended to be aggregator.create_data_products).
        """
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((UDP_IP, UDP_PORT))
        
        while not self.quit_data_thread:
            new_chunk = Chunk(sock, 4097)
            # 4097 chosen here so that the data is just an array of one array.
            # Aggregator should be changed to deal with multiple channels.
            self.publish_func(new_chunk.data[0])
            # This is where data will be pushed to aggregator, and in turn to subscribers.
            # Intended aggregator function is create_data_products_udp



class KatcpCatcher():
    def __init__(self, proc_func, bufname, roachip='roach'):
        self.bufname = bufname
        self.data_thread = None
        self.proc_func = proc_func
        from corr.katcp_wrapper import FpgaClient
        self.data_thread_r = FpgaClient(roachip, timeout=0.1)
        t1 = time.time()
        timeout = 10
        while not self.data_thread_r.is_connected():
            if (time.time() - t1) > timeout:
                raise Exception("Connection timeout to roach")
            time.sleep(0.1)
            
        self.last_addr = 0


    def start_data_thread(self):
        if self.data_thread:
            self.quit_data_thread = True
            self.data_thread.join(1.0)
            self.data_thread = None
        self.quit_data_thread = False
        self.data_thread = threading.Thread(target=self._cont_read_data, args=())
        # IMPORTANT - where cont_read_data comes in
        self.data_thread.daemon = True
        self.data_thread.start()
        
    def _proc_raw_data(self, data, addr):
        if addr - self.last_addr > 8192:
            print "skipped:", addr, self.last_addr, (addr - self.last_addr)
        self.last_addr = addr 
        data = np.fromstring(data, dtype='>i2').astype('float32').view('complex64')
        self.pxx = (np.abs(np.fft.fft(data.reshape((-1, 1024)), axis=1)) ** 2).mean(0)
        
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
            data = r.read(bram, 4 * 2 ** 12)
            self.proc_func(data, addr)
            # IMPORTANT - where proc_func comes in
            # coord passes self.aggregator.proc_raw_data as the proc_func here.
            
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
            
