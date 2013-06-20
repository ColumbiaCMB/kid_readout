#! /usr/bin/python

import time
import pylab as pl
from kid_readout.utils.single_pixel import SinglePixelHeterodyne

sp_readout = SinglePixelHeterodyne()
#sp_readout.initialize()
channel = 2**8
sp_readout.setChannel(channel, amp=-3)
dout, addrs = sp_readout.getData(nread=4)

#pl.plot(dout.real, label='real')
pl.plot(dout.imag, label='imag')
pl.legend()
pl.show()

#pl.clf()
#pl.plot(addrs, 'o')
#pl.show()


