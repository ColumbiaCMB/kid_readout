import Pyro4
import kid_readout.utils.mini_aggregator
import kid_readout.utils.catcher
import yappi

ns = Pyro4.naming.locateNS()

try:
    uri = ns.lookup("minicoord")
    proxy = Pyro4.Proxy(uri)
    proxy.quit()
    ns.remove("minicoord")
    print "removed old minicoord"
except:
    pass

class MiniCoordinator():
    def __init__(self):
        
        self.miniagg = kid_readout.utils.mini_aggregator.MiniAggregator()
        self.catcher = kid_readout.utils.catcher.DemultiplexCatcher(self.miniagg.create_data_products_debug)
        
        self.catcher.start_data_thread()
        
    def set_channel_ids(self, ids):
        return self.catcher.set_channel_ids(ids)
        
    def get_data(self, data_request):
        return self.miniagg.get_data(data_request)


yappi.start()
minicoord = MiniCoordinator()

minicoord.set_channel_ids([(i * 100) / 2 + 3 for i in range(1, 11)])
for i in range(1000):
    minicoord.get_data(10)
    
yappi.print_stats()
