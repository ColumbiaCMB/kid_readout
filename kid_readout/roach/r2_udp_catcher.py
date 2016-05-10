import numpy as np
import struct
import socket
from contextlib import closing
import time
import cPickle

def get_udp_packets(ri,npkts,addr=('10.0.0.1',55555)):
    ri.r.write_int('txrst',2)
    
    with closing(socket.socket(socket.AF_INET,socket.SOCK_DGRAM)) as s:
        s.bind(addr)
        s.settimeout(0)
        nstale = 0
        try:
            while s.recv(2000):
                nstale +=1
            if nstale:
                print "flushed",nstale,"packets"
        except Exception as e:
            pass
        s.settimeout(1)
        
        ri.r.write_int('txrst',0)
        pkts = []
        while len(pkts) < (npkts + 1):
            pkt = s.recv(2000)
            if pkt:
                pkts.append(pkt)
            else:
                print "breaking"
                break

    return pkts
    
def get_udp_data(ri,npkts,nchans,addr=('10.0.0.1',55555)):
    pkts = get_udp_packets(ri, npkts, addr=addr)
    darray, seqnos, num_bad_pkts, num_dropped_pkts = decode_packets(pkts,nchans)
    print "bad ", num_bad_pkts
    print "dropped ", num_dropped_pkts
    return darray,seqnos


def decode_packets(plist,nchans):
    assert(nchans>0)
    plist = plist[1:]
    cntr_total = 2**32
    nfft2 = 2**14 / 2
    chns_per_pkt = 256
    pkt_counter_step = nfft2 * chns_per_pkt / nchans
    max_num_pkts = cntr_total / pkt_counter_step
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
            k = (pkt_addr - start_addr) / pkt_counter_step + n*max_num_pkts
            if k >= npkts:
                break
            si = k * chns_per_pkt
            sf = (k + 1) * chns_per_pkt
            data[si:sf] = 1j*np.conj(all_data[:-1].view('<i2').astype('float32').view('complex64'))
            packet_counter[k] = pkt_addr

        num_bad_pkts += num_lost
        data = data.reshape((-1,nchans))
        num_dropped_pkts = np.sum(np.isnan(packet_counter)) - (num_bad_pkts - num_lost)
    return data, packet_counter, num_bad_pkts, num_dropped_pkts

def get_first_packet_index(plist):
    start = 0
    for pkt in plist:
        if len(pkt) != 1028:
            start += 1
            continue
        return start
    return len(plist)



ptype = np.dtype([('idle','>i2'),
                  ('idx', '>i2'),
                ('stream', '>i2'),
                ('chan', '>i2'),
                ('mcntr', '>i4')])

hdr_fmt = ">4HI"
hdr_size = struct.calcsize(hdr_fmt)
pkt_size = hdr_size + 1024
null_pkt = "\x00"*1024
def decode_packets_orig(plist,streamid,chans,nfft,pkts_per_chunk = 16,capture_failures=False):
    nchan = chans.shape[0]    
    mcnt_inc = nfft*2**12/nchan    
    next_seqno = None
    mcnt_top = 0
    dset = []
    mcntoff = None
    last_mcnt_ovf = None
    seqnos = []
    nextseqnos = []
    chan0 = None
    for pnum,pkt in enumerate(plist):
        if len(pkt) != pkt_size:
            print "got packet size",len(pkt), "expected",pkt_size
            continue
        pidle,pidx,pstream,pchan,pmcnt = struct.unpack(hdr_fmt,pkt[:hdr_size])
        #print pmcnt
        if pstream != streamid:
            print "got stream id",pstream,"expected",streamid
            continue            
        if next_seqno is None:
            mcnt_top = 0
            last_mcnt_ovf = pmcnt
        else:
            if pmcnt < mcnt_inc:
                if last_mcnt_ovf != pmcnt:
                    print "detected mcnt overflow",last_mcnt_ovf,pmcnt,pidx,next_seqno,(mcnt_top/2**32),pnum,mcntoff
                    mcnt_top += 2**32
                    last_mcnt_ovf = pmcnt
                else:
#                    print "continuation of previous mcnt overflow",pidx
                    pass
            else:
                last_mcnt_ovf = None
        chunkno,pmcntoff = divmod(pmcnt+mcnt_top,mcnt_inc)
        #print chunkno,pmcnt,pmcntoff,pidx
        seqno = (chunkno)*pkts_per_chunk + pidx
        #print seqno
        seqnos.append(seqno)
        nextseqnos.append(next_seqno)
        if next_seqno is None:
            chan0 = pchan
#            print "found first packet",seqno,pidx
            next_seqno = seqno
            mcntoff = pmcntoff
#            print pchan
        if mcntoff != pmcntoff:
            print "mcnt offset jumped. Was",mcntoff,"now",pmcntoff,"dropping.."
            continue
        if pchan != chan0:
            print "warning! channel id changed from",chan0,"to",pchan 
        if seqno - next_seqno < 0:
            print "seqno diff",(seqno-next_seqno),seqno,next_seqno
            continue # trying to go backwards
        if seqno == next_seqno:
            dset.append(pkt[hdr_size:])
            next_seqno += 1
        else:
            print "sequence number skip, expected:",next_seqno,"got",seqno,"inserting",(seqno-next_seqno),"null packets",pnum,pidx
            if capture_failures: #seqno-next_seqno == 32768:
                print "caught special case, writing to disk"
                fname = time.strftime("udp_skip_%Y-%m-%d_%H%M%S.pkl")
                fh = open(fname,'w')
                cPickle.dump(dict(plist=plist,dset=dset,pnum=pnum,pkt=pkt,streamid=streamid,chans=chans,nfft=nfft),
                             fh,cPickle.HIGHEST_PROTOCOL)
                fh.close()
                print "wrote data to:",fname
            for k in range(seqno - next_seqno+1):
                dset.append(null_pkt)
                next_seqno += 1
    dset = ''.join(dset)
    ns = (len(dset)//(4*nchan))
    dset = dset[:ns*(4*nchan)]
    darray = np.fromstring(dset,dtype='>i2').astype('float32').view('complex64')
    darray.shape = (ns,nchan)
    shift = np.flatnonzero(chans==(chan0))[0] - (nchan-1)
    darray = np.roll(darray,shift,axis=1)
    return darray,np.array(seqnos)
