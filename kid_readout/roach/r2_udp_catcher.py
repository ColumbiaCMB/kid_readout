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
        except:
            pass
        s.settimeout(1)
        
        ri.r.write_int('txrst',0)
        pkts = []
        while len(pkts) < npkts:
            pkt = s.recv(2000)
            if pkt:
                pkts.append(pkt)
            else:
                print "breaking"
                break

    return pkts
    
def get_udp_data(ri,npkts,nchans,addr=('10.0.0.1',55555)):
    pkts = get_udp_packets(ri, npkts, addr=addr)
    darray,seqnos = decode_packets(pkts,nchans)
    return darray,seqnos


def decode_packets(plist,nchans):
    packet_counter = []
    raw_data = []
    for pkt in plist:
        try: 
            all_data = np.fromstring(pkt,'<u4')
        except ValueError:
            #could put some code to avoid crashes here. ie just append zeros(257)
            print "got bad packet"
            continue 
        if all_data.shape[0] != 257:
            print "got weird packet",all_data.shape
            continue 
        raw_data.append(all_data[:-1])
        packet_counter.append(all_data[-1])
    data = np.hstack(raw_data)
    data = data.view('<i2').astype('float32').view('complex64')
    data = data.reshape((-1,nchans))
    packet_counter = np.array(packet_counter)
    return data,packet_counter

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
