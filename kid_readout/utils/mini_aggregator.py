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
        
        self.request = False
        self.ready = False
        self.last_product = None
        self.lock = threading.Lock()
        
        
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
        
        self.lock.acquire()
        self.request = True
        self.lock.release()
        # Acquires lock, asks for data, releases lock.
        
        while ii < data_request:
            self.lock.acquire()
            
            if self.ready == True:
                data_to_send.append(self.last_product)
                self.ready = False
                ii += 1
                # If new data is ready, appends it to the data_to_send.
                # Resets ready to false --> gets set to true if more data written.
                # Increments ii up towards data request.
                
            if ii == data_request:
                self.request = False
                # If ii == data_request, get data is doen, so request gets set to false/
                # This needs to happen here (rather than after the while loop) so that the function has the lock.
         
            self.lock.release()
            # After going through the while loop, the lock is released.
       
        return data_to_send
        # After all the data has been gathered, data_to_send is returned.
    
    
    def create_data_products_debug(self, packet):
        self.lock.acquire()
        
        if self.request == True:
            # Does stuff with the data only if request is true.
            
            data_list = []
            for i in range(len(packet)):
                data_product = dict(type='power spectrum', data=packet[i]['data'], clock=packet[i]['clock'],
                                   channel_id=packet[i]['channel_id'], addr=packet[i]['addr'], index=packet[i]['index'])
                data_list.append(data_product)
            self.last_product = data_list
            self.ready = True
            # ready is set to true when last_product is renewed.
            # ready gets set to false when get_data reads last_product.
            
        self.lock.release()
    
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
