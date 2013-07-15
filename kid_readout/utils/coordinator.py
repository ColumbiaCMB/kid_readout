import numpy as np
import time
import sys
import single_pixel
import threading
import catcher
import aggregator
import netcdf_writer
import Pyro4
import collections as col

class Coordinator(single_pixel.SinglePixelBaseband):
    def __init__(self, wafer=0, roachip='roach', adc_valon=None):
        single_pixel.SinglePixelBaseband.__init__(self, roach=None, wafer=wafer, roachip=roachip,
                                                  adc_valon=adc_valon)
        
        self.writer = netcdf_writer.NetCDFWriter(parent=self)
        self.aggregator = aggregator.Aggregator(parent=self, writer=self.writer)
        # self.catcher = catcher.KatcpCatcher(proc_func=self.aggregator.proc_raw_data, bufname=self.bufname, roachip=roachip)
        # self.catcher = catcher.UDPCatcher(publish_func=self.aggregator.create_data_products_udp, bufname=self.bufname, roachip=roachip)
        self.catcher = catcher.PacketCatcher(publish_func=self.aggregator.create_data_products_udp, bufname=self.bufname, roachip=roachip)
        
    def start_data_thread(self):
        self.catcher.start_data_thread()
    def subscribe_uri(self, uri, data_products):
        
        self.aggregator.subscribe_uri(uri, data_products)
