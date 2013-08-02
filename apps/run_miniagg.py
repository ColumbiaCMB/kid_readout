import Pyro4
import kid_readout.utils.mini_aggregator
import kid_readout.utils.catcher

ns = Pyro4.naming.locateNS()

try:
    uri = ns.lookup("miniagg")
    proxy = Pyro4.Proxy(uri)
    proxy.quit()
    ns.remove("miniagg")
    print "removed old miniagg"
except:
    pass

miniagg = kid_readout.utils.mini_aggregator.MiniAggregator()
catcher = kid_readout.utils.catcher.DemultiplexCatcher(miniagg.create_data_products_debug, bufname='fill')

catcher.start_data_thread()


daemon = Pyro4.Daemon()
uri = daemon.register(miniagg)
ns.register("miniagg", uri)

daemon.requestLoop()


