"""
Processing pipeline:
  * Capture UDP packets
    * (Buffer)
  * Decode int16 -> float32, recover packet sequence number
  * Demodulate
    * (Buffer)
  * Filter
  * Write to disk
  
Usage:
  * Do sweeps of resonators
  * Set tones to resonant frequencies
  * Start streaming processing pipeline for as long as desired
"""

import numpy as np
import socket
from contextlib import closing
import multiprocessing as mp
import time
import ctypes
from Queue import Empty as EmptyException
from kid_readout.roach import demodulator

pkt_size = 4100
data_ctype = ctypes.c_uint8
data_dtype = np.uint8
sequence_num_ctype = ctypes.c_uint32
counter_dtype = np.uint32
chns_per_pkt = 1024
samples_per_packet = 1024

class ReadoutPipeline:
    def __init__(self, nchans, num_data_buffers=4, num_packets_per_buffer=2 ** 12, output_size=2 ** 20,
                 host_address=('10.0.0.1',55555)):
        packet_buffer_size = pkt_size * num_packets_per_buffer
        self.num_data_buffers = num_data_buffers
        self.packet_data_buffers = [mp.Array(data_ctype, packet_buffer_size) for b in range(num_data_buffers)]

        demodulated_buffer_size = num_packets_per_buffer*samples_per_packet*np.dtype(np.complex64).size
        self.demodulated_data_buffers = [mp.Array(ctypes.c_uint8, demodulated_buffer_size) for b in range(num_data_buffers)]

        self.real_time_data_buffer = mp.Array(ctypes.c_uint8, output_size*np.dtype(np.complex64).size)

        # im not sure we are using this correctly. should there be a lock somewhere?
        # or does it not matter because only one process gets it
        self._sequence_num_buffer = mp.Array(sequence_num_ctype, output_size)
        self.sequence_num = np.frombuffer(self._sequence_num_buffer.get_obj(), dtype=counter_dtype)
        self.sequence_num[:] = 0

        self.capture_status = mp.Array(ctypes.c_char, 32)
        self.demodulate_status = mp.Array(ctypes.c_char, 32)

        self._num_bad_packets = mp.Value(ctypes.c_uint)
        self.num_bad_packets = self._num_bad_packets.get_obj()
        self.num_bad_packets.value = 0

        self.packet_input_queue = mp.Queue()
        self.packet_output_queue = mp.Queue()
        self.demodulated_input_queue = mp.Queue()
        self.demodulated_output_queue = mp.Queue()

        for i in range(num_data_buffers):
            self.packet_input_queue.put(i)
            self.demodulated_input_queue.put(i)

        self.process_data = DecodePacketsAndDemodulateProcess(packet_data_buffers=self.packet_data_buffers,
                                                              num_packets_per_buffer=num_packets_per_buffer,
                                                              packet_output_queue=self.packet_output_queue,
                                                              packet_input_queue=self.packet_input_queue,
                                                              demodulated_data_buffers=self.demodulated_data_buffers,
                                                              demodulated_input_queue=self.demodulated_input_queue,
                                                              sequence_num_buffer=self._sequence_num_buffer,
                                                              output_size=output_size,
                                                              nchans=nchans, status = self.demodulate_status)

        self.read_data = CapturePacketsProcess(packet_data_buffers=self.packet_data_buffers,
                                               num_packets_per_buffer=num_packets_per_buffer,
                                               packet_input_queue=self.packet_input_queue,
                                               packet_output_queue=self.packet_output_queue,
                                               bad_packets_counter=self._num_bad_packets,
                                               host_address=host_address, status = self.capture_status)

    def close(self):
        self.packet_input_queue.put(None)
        self.packet_output_queue.put(None)
        self.read_data.child.join()
        self.process_data.child.join()


class DecodePacketsAndDemodulateProcess:
    def __init__(self, packet_data_buffers, demodulated_data_buffers, num_packets_per_buffer,
                 packet_input_queue, packet_output_queue,
                 demodulated_input_queue, sequence_num_buffer, output_size,
                 nchans, status):
        self.output_index = 0
        self.packet_data_buffers = packet_data_buffers
        self.demodulated_data_buffers = demodulated_data_buffers
        self.num_packets_per_buffer = num_packets_per_buffer
        self.packet_input_queue = packet_input_queue
        self.packet_output_queue = packet_output_queue
        self.demodulated_input_queue = demodulated_input_queue
        self.sequence_num_buffer = sequence_num_buffer
        self.output_size = output_size
        self.nchans = nchans
        self.status = status
        self.status.value = "not started"
        self.child = mp.Process(target=self.run)
        self.child.start()

    def run(self):
        self.demodulator = demodulator.StreamDemodulator(tone_bins=self.tone_bins,phases=self.phases,
                                                         tone_nsamp=self.tone_nsamp, fft_bins=self.fft_bins,
                                                         nfft=self.nfft, )
        while True:
            try:
                process_me = self.packet_output_queue.get_nowait()
            except EmptyException:
                self.status.value = "waiting"
                time.sleep(0.01)
                continue
            if process_me is None:
                break
            else:
                self.status.value = "blocked"
                output_to = self.demodulated_input_queue.get()

                with self.packet_data_buffers[process_me].get_lock(), self.demodulated_data_buffers[output_to].get_lock():
                    self.status.value = "processing"
                    packets = np.frombuffer(self.packet_data_buffers[process_me].get_obj(), dtype=data_dtype)
                    packets.shape=(self.num_packets_per_buffer, pkt_size)
                    pkt_counter = packets.view(counter_dtype)[:,-1]

#                    contiguous = np.all(np.diff(pkt_counter)==1)

                    demod_data = np.frombuffer(self.demodulated_data_buffers[output_to].get_obj(), dtype=np.complex64)

                    raw_data = packets[:,:-4].view('<i2').astype(np.float32).view(np.complex64)
                    raw_data = raw_data.reshape((-1,self.nchans))


                    demod_data = np.transpose(demod_data.reshape((-1,self.nchans,2)),axes=(2,0,1))

                    # Decode packets
                    for k in range(self.num_packets_per_buffer):
                        si = k * chns_per_pkt
                        sf = (k + 1) * chns_per_pkt
                        demod_data[si:sf] = packets[k,:-4].view('<i2').astype('float32').view('complex64')
                        self.sequence_num_buffer[self.output_index] = pkt_counter[k]
                        self.output_index += 1
                        self.output_index %= self.output_size
                    demod_data = demod_data.reshape((-1,self.nchans))

#                    data = r2.demodulate_stream(data, pkt_counter)
                    #print "decode ", timeit.default_timer() - t0
                self.demodulated_output_queue.put(output_to)
                self.packet_input_queue.put(process_me)
        self.status.value = "exiting"
        return None


class CapturePacketsProcess:
    def __init__(self, packet_data_buffers, num_packets_per_buffer, packet_input_queue, packet_output_queue,
                 bad_packets_counter, host_address,status):
        self.packet_data_buffers = packet_data_buffers
        self.num_packets_per_buffer = num_packets_per_buffer
        self.packet_input_queue = packet_input_queue
        self.packet_output_queue = packet_output_queue
        self.bad_packets_counter = bad_packets_counter
        self.host_address = host_address
        self.status = status
        self.status.value = "starting"
        self.child = mp.Process(target=self.run)
        self.child.start()

    def run(self):
        with closing(socket.socket(socket.AF_INET,socket.SOCK_DGRAM)) as s:
            s.bind(self.host_address)
            s.settimeout(1)
            while True:
                try:
                    process_me = self.packet_input_queue.get_nowait()
                except EmptyException:
                    self.status.value = "blocked"
                    time.sleep(0.005)
                    continue
                if process_me is None:
                    break
                else:
                    with self.packet_data_buffers[process_me].get_lock():
                        self.status.value = "processing"
                        #t0 = timeit.default_timer()
                        packet_buffer = np.frombuffer(self.packet_data_buffers[process_me].get_obj(), dtype=data_dtype)
                        packet_buffer.shape=(self.num_packets_per_buffer, pkt_size)
                        i = 0
                        while i < self.num_packets_per_buffer:
                            pkt = s.recv(5000)
                            if len(pkt) == pkt_size:
                                packet_buffer[i,:] = np.frombuffer(pkt, dtype=data_dtype)
                                i += 1
                            else:
                                print "got a bad packet"
                                self.bad_packets_counter.value += 1
                        #print "read: ", timeit.default_timer() - t0
                    self.packet_output_queue.put(process_me)
        self.status.value = "exiting"
        return None
