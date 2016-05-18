cimport numpy as np
import numpy as np


def get_first_packet_index(plist):
    start = 0
    for pkt in plist:
        if len(pkt) != 1028:
            start += 1
            continue
        return start
    return len(plist)


def decode_packets_fast(plist,nchans):
    assert(nchans>0)
    cdef long cntr_total
    cdef unsigned int nfft2, chns_per_pkt, pkt_counter_step, max_num_pkts, npkts
    plist = plist[1:]
    cntr_total = 2**32
    nfft2 = 2**14 / 2
    chns_per_pkt = 256
    pkt_counter_step = nfft2 * chns_per_pkt // nchans
    max_num_pkts = cntr_total // pkt_counter_step
    npkts = len(plist)

    start_ind = get_first_packet_index(plist)
    end_ind = npkts - get_first_packet_index(plist[::-1])
    num_lost = start_ind + npkts - end_ind
    plist = plist[start_ind:end_ind]

    packet_counter = np.zeros(npkts, dtype='uint32')
    data = np.empty(npkts*chns_per_pkt, dtype='complex64')
    data.fill(np.nan+1j*np.nan)

    cdef unsigned int start_addr, pkt_addr, n
    cdef unsigned long k

    if len(plist) == 0:
        num_bad_pkts = npkts
        num_dropped_pkts = 0
    else:
        start_addr = np.fromstring(plist[0],'<u4')[-1]
        num_bad_pkts = 0
        n = 0
        was_below = False
        for pkt in plist:
            if len(pkt) != 1028:
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