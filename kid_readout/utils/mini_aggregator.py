import numpy as np
import time
import sys
import threading
import Pyro4
import collections as col


class MiniAggregator():
    def __init__(self):
        self.subscriptions = {'power spectrum': []}
        self.subscribers = {}  # maps uri to subscriber
        self.last_addr = 0
        self.is_ready = False
        self.last_product = None
        self.condition = threading.Condition()
        
        
    def subscribe_uri(self, uri, data_products):
        if self.subscribers.has_key(uri):
            print "already subscribed!", uri
            return
        subscriber = Pyro4.Proxy(uri)
        self.subscribers[uri] = subscriber
        for data_product in data_products:
            self.subscriptions[data_product].append(subscriber)
            
    def publish_test(self, data_product):
        
        for i in range(len(data_product)):
            print data_product[i]['channel_id']
            print data_product[i]['index']
            print data_product[i]['data']
        # Used for debugging

    def get_data(self, data_request):
        data_to_send = []
        ii = 0
        self.condition.acquire()
        while ii < data_request:
            if self.is_ready == True:
                data_to_send.append(self.last_product)
                self.is_ready = False
                ii += 1
            self.condition.wait()
        self.condition.release()
            
        return data_to_send
    # Check out http://docs.python.org/2/library/threading.html#condition-objects
    # Looks promising to create the wait function correctly.
    
    
    def create_data_products_debug(self, packet):
        self.condition.acquire()
        data_list = []
        for i in range(len(packet)):
            data_product = dict(type='power spectrum', data=packet[i]['data'], clock=packet[i]['clock'],
                               channel_id=packet[i]['channel_id'], addr=packet[i]['addr'], index=packet[i]['index'])
            data_list.append(data_product)
        self.last_product = data_list
        self.condition.notify()
        self.is_ready = True
        self.condition.release()
    
    def create_data_products_dict(self, packet):
        pxx_list = []
        for i in range(len(packet)):
            raw_data = packet[i]['data']
            pxx = (np.abs(np.fft.fft(raw_data)) ** 2)
            pxx_product = dict(type='power spectrum', data=pxx, clock=packet[i]['clock'],
                               channel_id=packet[i]['channel_id'], addr=packet[i]['addr'], index=packet[i]['index'])
            pxx_list.append(pxx_product)
        # self.publish_test(pxx_list)
        self.publish_test(pxx_list)
        
    def publish(self, data_product):
        # pass the data products on to the appropriate subscribers
        subscription_list = self.subscriptions[data_product['type']]  # list of subscribers interested in this data product
        print "publishing pxx to", subscription_list
        for subscriber in subscription_list:
            subscriber.handle(data_product)
