import socket
from contextlib import closing
import multiprocessing as mp
from multiprocessing import Process, Queue
import time
import ctypes
from Queue import Empty as EmptyException


pkt_size = 4100
data_ctype = ctypes.c_uint8
data_dtype = np.uint8
counter_ctype = ctypes.c_uint32
counter_dtype = np.uint32
chns_per_pkt = 1024

class ReadoutPipeline:
    def __init__(self, nchans, num_data_buffers=4, npkts_per_buffer=2**12, output_size=2**20):
        buffer_size = pkt_size*npkts_per_buffer
        self.num_data_buffers = num_data_buffers
        self.data_buffers = [mp.Array(data_ctype, buffer_size) for b in range(num_data_buffers)]
        
        # im not sure we are using this correctly. should there be a lock somewhere?
        # or does it not matter because only one process gets it
        self._counter_buffer = mp.Array(counter_ctype, output_size)
        self.counter = np.frombuffer(self._counter_buffer.get_obj(), dtype=counter_dtype)
        self.counter[:] = 0
        
        self._nbad_pkts = mp.Value(ctypes.c_uint)
        self.nbad_pkts = self._nbad_pkts.get_obj()
        self.nbad_pkts = 0
        
        self.input_queue = mp.Queue()
        self.output_queue = mp.Queue()
        for i in range(num_data_buffers):
            self.input_queue.put(i)
            
        self.process_data = DecodeDataProcess(data_buffers=self.data_buffers,
                                              npkts_per_buffer=npkts_per_buffer,
                                              input_queue=self.output_queue,
                                              output_queue=self.input_queue,
                                              counter_buffer=self._counter_buffer,
                                              output_size=output_size, 
                                              nchans=nchans)
        
        self.read_data = ReadDataProcess(data_buffers=self.data_buffers,
                                         npkts_per_buffer=npkts_per_buffer,
                                         input_queue=self.input_queue,
                                         output_queue=self.output_queue, 
                                         nbad_pkts=self._nbad_pkts)   
        return None

    def close(self):
        self.input_queue.put(None)
        self.output_queue.put(None)
        self.read_data.child.join()
        self.process_data.child.join()


class DecodeDataProcess:
    def __init__(self, data_buffers, npkts_per_buffer, input_queue, output_queue, counter_buffer, output_size,
                 nchans):
        self.out_ind = 0
        self.child = mp.Process(target=self.decode_data,
                                kwargs=dict(data_buffers=data_buffers, 
                                            npkts_per_buffer=npkts_per_buffer, 
                                            input_queue=input_queue, 
                                            output_queue=output_queue, 
                                            counter_buffer=counter_buffer,
                                            output_size=output_size, 
                                            nchans=nchans))
        
        self.child.start()
        return None

    def decode_data(self, data_buffers, npkts_per_buffer, input_queue, output_queue, counter_buffer, output_size,
                    nchans):
        while True:
            try:
                process_me = input_queue.get_nowait()
            except EmptyException:
                time.sleep(0.01)
                continue
            if process_me is None:
                break
            else:
                with data_buffers[process_me].get_lock():
                    #t0 = timeit.default_timer()
                    packets = np.frombuffer(data_buffers[process_me].get_obj(), dtype=data_dtype)
                    packets.shape=(npkts_per_buffer, pkt_size)
                    pkt_counter = packets.view(counter_dtype)[:,-1]
                    
                    # Decode packets
                    data = np.empty(npkts_per_buffer*chns_per_pkt, dtype='complex64')
                    for k in range(npkts_per_buffer):
                        si = k * chns_per_pkt
                        sf = (k + 1) * chns_per_pkt
                        data[si:sf] = packets[k,:-4].view('<i2').astype('float32').view('complex64')
                        counter_buffer[self.out_ind] = pkt_counter[k]
                        self.out_ind += 1
                        self.out_ind %= output_size
                    data = data.reshape((-1,nchans))
                    data = r2.demodulate_stream(data, pkt_counter)
                    #print "decode ", timeit.default_timer() - t0
                output_queue.put(process_me)
        return None
        
        
class ReadDataProcess:
    def __init__(self, data_buffers, npkts_per_buffer, input_queue, output_queue, nbad_pkts):
        self.child = mp.Process(target=self.read_data,
                                kwargs=dict(data_buffers=data_buffers, 
                                            npkts_per_buffer=npkts_per_buffer, 
                                            input_queue=input_queue,
                                            output_queue=output_queue,
                                            nbad_pkts=nbad_pkts))
        self.child.start()
        return None

    def read_data(self, data_buffers, npkts_per_buffer, input_queue, output_queue, nbad_pkts, 
                  addr=('10.0.0.1',55555)):
        with closing(socket.socket(socket.AF_INET,socket.SOCK_DGRAM)) as s:
            s.bind(addr)
            s.settimeout(1)
            while True:
                try:
                    process_me = input_queue.get_nowait()
                except EmptyException:
                    time.sleep(0.005)
                    continue
                if process_me is None:
                    break
                else:
                    with data_buffers[process_me].get_lock():
                        #t0 = timeit.default_timer()
                        packet_buffer = np.frombuffer(data_buffers[process_me].get_obj(), dtype=data_dtype)
                        packet_buffer.shape=(npkts_per_buffer, pkt_size)
                        i = 0
                        while i < npkts_per_buffer:
                            pkt = s.recv(5000)
                            if len(pkt) == pkt_size:
                                packet_buffer[i,:] = np.frombuffer(pkt, dtype=data_dtype)
                                i += 1
                            else:
                                print "got a bad packet"
                                nbad_pkts += 1
                        #print "read: ", timeit.default_timer() - t0
                    output_queue.put(process_me)   
        return None
