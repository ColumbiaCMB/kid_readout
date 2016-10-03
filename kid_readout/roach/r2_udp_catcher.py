import numpy as np
import socket
from contextlib import closing

import logging
logger = logging.getLogger(__name__)

try:
    # must compile cython code by running: python setup.py build_ext --inplace
    import decode
    have_decode=True
except ImportError:
    have_decode=False


def get_udp_packets(ri,npkts,addr=('10.0.0.1',55555)):
    ri.r.write_int('txrst',2)

    with closing(socket.socket(socket.AF_INET,socket.SOCK_DGRAM)) as s:
        s.bind(addr)
        s.settimeout(0)
        nstale = 0
        try:
            while s.recv(5000):
                nstale +=1
            if nstale:
                print "flushed",nstale,"packets"
        except Exception as e:
            pass
        s.settimeout(1)
        
        ri.r.write_int('txrst',0)
        pkts = []
        while len(pkts) < (npkts + 1):
            try:
                pkt = s.recv(5000)
            except socket.timeout:
                logger.error("Socket timeout waiting for packets from ROACH. This probably means the GbE is jammed. "
                             "Attempting to restart GbE")
                ri.r.write_int('txrst',1)
                ri.r.write_int('txrst',0)
            else:
                if pkt:
                    pkts.append(pkt)
                else:
                    logger.warning("Timed out waiting for packets from the ROACH")
                    break

    return pkts


def get_udp_data(ri,npkts,nchans,addr=('10.0.0.1',55555), verbose=False, fast=False):
    pkts = get_udp_packets(ri, npkts, addr=addr)
    if fast:
        darray, seqnos, num_bad_pkts, num_dropped_pkts = decode.decode_packets_fast(pkts,nchans)
    else:
        darray, seqnos, num_bad_pkts, num_dropped_pkts = decode_packets(pkts,nchans,ri.fpga_cycles_per_filterbank_frame)
    if num_bad_pkts or num_dropped_pkts:
        logger.warning("Detected %d bad and %d dropped packets. Something is likely misconfigured" % (num_bad_pkts,num_dropped_pkts))
    if verbose:
        print "bad ", num_bad_pkts
        print "dropped ", num_dropped_pkts
    return darray,seqnos


def decode_packets(plist,nchans,clocks_per_filterbank_frame):
    assert(nchans>0)
    plist = plist[1:]
    cntr_total = 2**32
    chns_per_pkt = 1024
    pkt_counter_step = clocks_per_filterbank_frame * chns_per_pkt // nchans
    max_num_pkts = cntr_total // pkt_counter_step
    npkts = len(plist)

    start_ind = get_first_packet_index(plist)
    end_ind = npkts - get_first_packet_index(plist[::-1])
    num_lost = start_ind + npkts - end_ind
    plist = plist[start_ind:end_ind]

    packet_counter = np.zeros(npkts, dtype='uint32')
    data = np.empty(npkts*chns_per_pkt, dtype='complex64')
    data.fill(np.nan+1j*np.nan)

    if len(plist) == 0:
        num_bad_pkts = npkts
        num_dropped_pkts = 0
    else:
        start_addr = np.fromstring(plist[0],'<u4')[-1]
        #stop_addr = np.fromstring(plist[-1],'<u4')[-1]
        #num_expected_pkts = (stop_addr - start_addr) / pkt_counter_step + 1
        #if npkts > max_num_pkts:
        #    num_expected_pkts += int(npkts / max_num_pkts) * max_num_pkts
        num_bad_pkts = 0
        n = 0
        was_below = False
        for pkt in plist:
            if len(pkt) != 4100:
                num_bad_pkts += 1
                continue
            all_data = np.fromstring(pkt,'<u4')
            pkt_addr = all_data[-1]
            if was_below and (pkt_addr >= start_addr):
                n += 1
                was_below = False
            if pkt_addr < start_addr:
                was_below = True
            k = (pkt_addr - start_addr) // pkt_counter_step + n*max_num_pkts
            if k >= npkts:
                break
            si = k * chns_per_pkt
            sf = (k + 1) * chns_per_pkt
            data[si:sf] = all_data[:-1].view('<i2').astype('float32').view('complex64')
            packet_counter[k] = pkt_addr

        num_bad_pkts += num_lost
        num_dropped_pkts = np.sum(np.isnan(data))/nchans - (num_bad_pkts - num_lost)
        data = data.reshape((-1,nchans))
    return data, packet_counter, num_bad_pkts, num_dropped_pkts

def get_first_packet_index(plist):
    start = 0
    for pkt in plist:
        if len(pkt) != 4100:
            start += 1
            continue
        return start
    return len(plist)

