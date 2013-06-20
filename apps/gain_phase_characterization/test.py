#! /usr/bin/python

import time
import pylab as pl
from kid_readout.utils.single_pixel import SinglePixelHeterodyne

sp_readout = SinglePixelHeterodyne()
#sp_readout.initialize()
channel = 2**2
sp_readout.set_channel(channel, amp=-2)
dout, addrs = sp_readout.get_data(nread=2)

#pl.plot(dout.real, label='real')
pl.plot(dout.imag, label='imag')
pl.legend()
pl.show()

#pl.clf()
#pl.plot(addrs, 'o')
#pl.show()


