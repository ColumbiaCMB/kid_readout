import numpy as np
import time
import sys
import threading
import socket
import collections as col


class DemultiplexCatcher():
    ''' Receives packets from UDP socket, 
    
    decodes from a character string to a namedtuple, checks order and pushes to aggregator.
    
    '''
    
    def __init__(self, publish_func, bufname, roachip='roach'):
        self.bufname = bufname
        self.data_thread = None
        self.publish_func = publish_func
        self.channel_ids = [0, 1, 2, 3, 4]
        # self.channel_ids = [1003, 2003, 3003, 4003, 5003]
        # Defaults used for testing.
        
        self.packet_counter = 0
        self.last_packet = None
        # Used for packet ordering.
        
        self.time_zero = 0
        self.zero_switch = 0
        
    def start_data_thread(self):
        if self.data_thread:
            self.quit_data_thread = True
            self.data_thread.join(1.0)
            self.data_thread = None
        self.quit_data_thread = False
        self.data_thread = threading.Thread(target=self._cont_read_data, args=("localhost", 12345, 2))
        # Used for debugging on localhost.
        # self.data_thread = threading.Thread(target=self._cont_read_data, args=("192.168.1.1", 12345, 10))
        # Using the port and IP startup_server runs on for now.
        self.data_thread.daemon = True
        self.data_thread.start()
        
    def set_channel_ids(self, ids): 
        self.channel_ids = ids
        self.start_data_thread()
        
    def decode(self, pkt):
    
        index = np.fromstring(pkt[:2], dtype='>i2')
        channel_id = np.fromstring(pkt[2:4], dtype='>i2')
        addr = np.fromstring(pkt[4:8], dtype='>u4')
        data_list = np.fromstring(pkt[8:], dtype='>i2').astype('float').view('complex')
        '''Read data as an array of 2byte integers --> convert to float --> view as
        complex pairs (real and imaginary)'''
        
        multiplex_packet = dict([('index', index[0]), ('channel_id', channel_id[0]), ('addr', addr[0]), ('data_list' , data_list)])
        return multiplex_packet
    
    def get_clock(self, packet, chan_number):
        
        if self.zero_switch == 0:
            self.time_zero = packet['addr']
            self.zero_switch = 1
        
        clock = np.arange((len(packet['data_list']) + (-len(packet['data_list']) % chan_number)) / chan_number)
        # Creates a clock array of length equal to one channel.
        # The -len(packet['data_list'])%chan_number is the amount that must be added to the length of the packet/chan to be evenly divisible.
        # Essentially just a round-up function.
        
        # Works for even channel numbers. However, a remainder problem here.
        # clock += ((packet['addr'] / 4096) * 4096 - (self.time_zero / 4096) * 4096) / chan_number + len(clock) * packet['index']
        
        
        clock += ((packet['addr'] / 4096) - (self.time_zero / 4096)) * len(clock) * 16 + len(clock) * packet['index']
        # 16 packets has a total clock length not as described in the commented-out equation. It has a length of 16*len(clock).
        # This equation takes the amount of 16-packet chunks since time zero (addr/4096 - timezero/4096) and multiplies it by 16 * clock_length.
        
        packet['clock'] = clock
        
    
    
    def check_order(self, packet, chan_number):
        # Packet order is checked, then packet is pushed to aggregator.
        
        fill_data = None
        if self.packet_counter % 16 == packet['index']:
            self.packet_counter += 1
            self.last_packet = packet
            self.demultiplex(packet, chan_number)
        else:
            if self.packet_counter > packet['index']:
                print "late packet tossed"
                # Throws away late packets.
                
            if self.packet_counter < packet['index']:
                
                # Creates filler packet and send it. Channel_id from last valid packet.
                # Gets correct index and data filled with 'NaN'.
                # Updates clock to increment from last packet.
                
                fill_array = np.empty(len(self.last_packet['data_list']))
                # Error if the first packet received is out of order: no last_packet.
                fill_array[0:] = 'nan'
                # Fills all channels of the filler packet with 'NaN'.
                
                while self.packet_counter < packet['index']:
                    # Loop runs until packet_counter = packet.index. Fills discrepancy with filler packets.
                    
                    fill_clock = np.arange(len(self.last_packet['clock']))
                    fill_clock += self.last_packet['clock'][-1] + 1
                    # Updates the clock of the filler based on the previous packet.
                  
                    new_filler = dict([('index', self.packet_counter), ('channel_id', self.last_packet['channel_id']),
                                       ('addr', self.last_packet['addr']), ('data_list' , fill_array), ('clock', fill_clock)])
                    self.demultiplex(new_filler, chan_number)
                    # Makes the filler packet and publishes it.
                    
                    self.last_packet = new_filler
                    self.packet_counter += 1
                    # Updates variables.
               
                
                self.last_packet = packet
                self.demultiplex(packet, chan_number)
                self.packet_counter += 1
                # Once the while loop terminates, the function operates normally.
                
    def demultiplex(self, packet, chan_number):
        
        
        packet_list = []
        chan_index = 0
        
        for i in range(chan_number):
            if i == 0:
                channel_id = packet['channel_id']
                chan_index = self.channel_ids.index(channel_id)
                chan_index += 1
            else:
                channel_id = self.channel_ids[chan_index % len(self.channel_ids)]
                chan_index += 1
            # These statements assign the correct channel_id to each demultiplexed packet.
            
            data = packet['data_list'][i::chan_number]
            # Does the demultiplexing of the data.
            
            demultiplex_packet = dict([('index', packet['index']), ('channel_id', channel_id), ('addr', packet['addr']),
                                       ('clock', packet['clock'][:len(data)]), ('data' , data)])
            packet_list.append(demultiplex_packet)
            # Makes a list of the demultiplexed packets that is pushed to the publish_func.
        self.publish_func(packet_list)
            
    
    def _cont_read_data(self, udp_ip, udp_port, chan_number):
        """
        Reads data from socket as a chunk. Passes it to a publishing function passed
        to UDPCatcher (intended to be aggregator.create_data_products).
        """
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((udp_ip, udp_port))
        
        while not self.quit_data_thread:
            raw_data = sock.recv(10000)
            packet = self.decode(raw_data)
            self.get_clock(packet, chan_number)
            self.check_order(packet, chan_number)
            # Packet formed from raw input, sent to ordering.


class PacketCatcher():
    ''' Receives packets from UDP socket, 
    
    decodes from a character string to a namedtuple, checks order and pushes to aggregator.
    
    '''
    
    def __init__(self, publish_func, bufname, roachip='roach'):
        self.bufname = bufname
        self.data_thread = None
        self.publish_func = publish_func
        
        self.packet_counter = 0
        self.last_packet = None
        # Used for packet ordering.
        
        self.time_zero = None
        self.zero_switch = 0
        
    def start_data_thread(self):
        if self.data_thread:
            self.quit_data_thread = True
            self.data_thread.join(1.0)
            self.data_thread = None
        self.quit_data_thread = False
        self.data_thread = threading.Thread(target=self._cont_read_data, args=("192.168.1.1", 12345, 1))
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
            # Demultiplexes data.
            
            
        if self.zero_switch == 0:
            self.time_zero = addr
            self.zero_switch = 1
            # time_zero can be reset to what is needed. For testing, it sets the first clock it receives to zero.
            
            
        clock = np.arange(len(data[0]))
        clock += ((addr / 4096) * 4096 + (1024 / chan) * index) - self.time_zero
        # Creates the clock, incrementing by one.
        
        packet = col.namedtuple('Packet', ['index', 'channel_id', 'clock', 'data'])
        my_packet = packet(index, channel_id, clock, data)
    
        return my_packet
    
    
    def order(self, packet):
        # Packet order is checked, then packet is pushed to aggregator.
        
        fill_data = None
        if self.packet_counter % 16 == packet.index[0]:
            self.packet_counter += 1
            self.last_packet = packet
            self.publish_func(packet)
        else:
            if self.packet_counter > packet.index[0]:
                pass;
                # Throws away late packets.
                
            if self.packet_counter < packet.index[0]:
                
                # Creates filler packet and send it. Channel_id from last valid packet.
                # Gets correct index and data filled with 'NaN'.
                # Updates clock to increment from last packet.
                
                fill_array = np.empty(len(self.last_packet.data[0]))
                fill_array[0:] = 'nan'
                
                fill_data = range(len(self.last_packet.data))
                for i in len(fill_data):
                    fill_data[i] = fill_array
                # Fills all channels of the filler packet with 'NaN' at the longest possible length of any channel.
                # Makes the filler data once for any number of filler packets.
                
                while self.packet_counter < packet.index[0]:
                    # Loop runs until packet_counter = packet.index. Fills discrepancy with filler packets.
                    fill_clock = np.arange(len(fill_data[0]))
                    fill_clock += last_packet.clock[-1] + 1
                    # Updates the clock of the filler based on the previous packet.
                
                    Packet = col.namedtuple('Packet', ['index', 'channel_id', 'clock', 'data'])    
                    new_filler = Packet(self.packet_counter, self.last_packet.channel_id, fill_clock, fill_data)
                    self.publish_func(new_filler)
                    # Makes the filler packet and publishes it.
                    
                    self.last_packet = new_filler
                    self.packet_counter += 1
                    # Updates variables.
               
                
                self.last_packet = packet
                self.publish_func(packet)
                self.packet_counter += 1
                # Once the while loop terminates, the function operates normally. 
    
    def _cont_read_data(self, UDP_IP, UDP_PORT, CHANNEL_NUMBER):
        """
        Reads data from socket as a chunk. Passes it to a publishing function passed
        to UDPCatcher (intended to be aggregator.create_data_products).
        """
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((UDP_IP, UDP_PORT))
        
        while not self.quit_data_thread:
            raw_data = sock.recv(10000)
            packet = self.decode(raw_data, CHANNEL_NUMBER)
            self.order(packet)
            # Packet formed from raw input, sent to ordering.

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
            # Where proc_func comes in
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
            
