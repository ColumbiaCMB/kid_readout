import Pyro4
import kid_readout.utils.coordinator

ns = Pyro4.naming.locateNS()

try:
    uri = ns.lookup("BasebandCoordinator")
    proxy = Pyro4.Proxy(uri)
    proxy.quit()
    ns.remove("BasebandCoordinator")
    print "removed old coordinator"
except:
    pass

coord = kid_readout.utils.coordinator.Coordinator()
daemon = Pyro4.Daemon()
uri = daemon.register(coord)
ns.register("BasebandCoordinator",uri)

daemon.requestLoop()

