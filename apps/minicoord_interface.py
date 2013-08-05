import Pyro4

ns = Pyro4.naming.locateNS()

uri = ns.lookup("minicoord")
c = Pyro4.Proxy(uri)

print 'Methods of minicoords are get_data(number_of_packets) and set_channel_ids([list_of_channel_ids])'
