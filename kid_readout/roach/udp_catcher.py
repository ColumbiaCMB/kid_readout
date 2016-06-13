import time
import struct
import socket
import logging
from contextlib import closing
import cPickle

import numpy as np

# TODO: verify that the log levels are correct here.
logger = logging.getLogger(__name__)


def get_udp_packets(ri, npkts, streamid, stream_reg='streamid', addr=('192.168.1.1', 12345)):
    ri.r.write_int(stream_reg, 0)
    with closing(socket.socket(socket.AF_INET, socket.SOCK_DGRAM)) as s:
        s.bind(addr)
        s.settimeout(0)
        nstale = 0
        try:
            while s.recv(2000):
                nstale += 1
            if nstale:
                logger.info("Flushed {} packets.".format(nstale))
        except socket.error:
            pass
        s.settimeout(1)
        ri.r.write_int(stream_reg, streamid)
        pkts = []
        while len(pkts) < npkts:
            pkt = s.recv(2000)
            if pkt:
                pkts.append(pkt)
            else:
                logger.warning("Did not receive UDP data.")
                break
        ri.r.write_int(stream_reg, 0)
    return pkts


def get_udp_data(ri, npkts, streamid, chans, nfft, stream_reg='streamid', addr=('192.168.1.1', 12345)):
    pkts = get_udp_packets(ri, npkts, streamid, stream_reg=stream_reg, addr=addr)
    darray, seqnos = decode_packets(pkts, streamid, chans, nfft)
    return darray, seqnos


ptype = np.dtype([('idle', '>i2'),
                  ('idx', '>i2'),
                  ('stream', '>i2'),
                  ('chan', '>i2'),
                  ('mcntr', '>i4')])

hdr_fmt = ">4HI"
hdr_size = struct.calcsize(hdr_fmt)
pkt_size = hdr_size + 1024
null_pkt = "\x00" * 1024


def decode_packets(plist, streamid, chans, nfft, pkts_per_chunk=16, capture_failures=False):
    nchan = chans.shape[0]
    mcnt_inc = nfft * 2 ** 12 / nchan
    next_seqno = None
    mcnt_top = 0
    dset = []
    mcntoff = None
    last_mcnt_ovf = None
    seqnos = []
    nextseqnos = []
    chan0 = None
    for pnum, pkt in enumerate(plist):
        if len(pkt) != pkt_size:
            logger.warning("Packet size is {} but expected {}.".format(len(pkt), pkt_size))
            continue
        pidle, pidx, pstream, pchan, pmcnt = struct.unpack(hdr_fmt, pkt[:hdr_size])
        if pstream != streamid:
            logger.warning("Stream id is {} but expected {}".format(pstream, streamid))
            continue
        if next_seqno is None:
            mcnt_top = 0
            last_mcnt_ovf = pmcnt
        else:
            if pmcnt < mcnt_inc:
                if last_mcnt_ovf != pmcnt:
                    message = "Detected mcnt overflow {} {} {} {} {} {} {}"
                    logger.info(message.format(last_mcnt_ovf, pmcnt, pidx, next_seqno, mcnt_top / 2 ** 32, pnum, mcntoff))
                    mcnt_top += 2 ** 32
                    last_mcnt_ovf = pmcnt
                else:
                    #                    print "continuation of previous mcnt overflow",pidx
                    pass
            else:
                last_mcnt_ovf = None
        chunkno, pmcntoff = divmod(pmcnt + mcnt_top, mcnt_inc)
        # print chunkno,pmcnt,pmcntoff,pidx
        seqno = (chunkno) * pkts_per_chunk + pidx
        # print seqno
        seqnos.append(seqno)
        nextseqnos.append(next_seqno)
        if next_seqno is None:
            chan0 = pchan
            #            print "found first packet",seqno,pidx
            next_seqno = seqno
            mcntoff = pmcntoff
        #            print pchan
        if mcntoff != pmcntoff:
            logger.warning("mcnt offset jumped: was {} and is now {} ... dropping ...".format(mcntoff, pmcntoff))
            continue
        if pchan != chan0:
            logger.warning("warning: channel id changed from {} to {}.".format(chan0, pchan))
        if seqno - next_seqno < 0:
            logger.warning("seqno diff: {} {} {}".format(seqno - next_seqno, seqno, next_seqno))
            continue  # trying to go backwards
        if seqno == next_seqno:
            dset.append(pkt[hdr_size:])
            next_seqno += 1
        else:
            message = "sequence number skip: expected {} and got {}; inserting {} null packets; {} {}"
            logger.warning(message.format(next_seqno, seqno, seqno - next_seqno, pnum, pidx))
            if capture_failures:  # seqno-next_seqno == 32768:
                fname = time.strftime("udp_skip_%Y-%m-%d_%H%M%S.pkl")
                logger.warning("caught special case, writing to disk: {}".format(fname))
                fh = open(fname, 'w')
                cPickle.dump(
                    dict(plist=plist, dset=dset, pnum=pnum, pkt=pkt, streamid=streamid, chans=chans, nfft=nfft),
                    fh, cPickle.HIGHEST_PROTOCOL)
                fh.close()
            for k in range(seqno - next_seqno + 1):
                dset.append(null_pkt)
                next_seqno += 1
    dset = ''.join(dset)
    ns = (len(dset) // (4 * nchan))
    dset = dset[:ns * (4 * nchan)]
    darray = np.fromstring(dset, dtype='>i2').astype('float32').view('complex64')
    darray.shape = (ns, nchan)
    shift = np.flatnonzero(chans == (chan0))[0] - (nchan - 1)
    darray = np.roll(darray, shift, axis=1)
    return darray, np.array(seqnos)
