from matplotlib import pyplot as plt
import numpy as np
import Pyro4


class SimpleViewer():
    def __init__(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.line = None
        
    def handle(self,data):
        if data['type'] == 'power spectrum':
            y = np.log10(data['data'])
            x = np.arange(len(y))
            if self.line:
                self.line.update_data(x,y)
            else:
                self.line, = self.ax.plot(x,y)
            self.fig.canvas.draw()
        
        
ns = Pyro4.naming.locateNS()

viewer = SimpleViewer()
daemon = Pyro4.Daemon()
uri = daemon.register(viewer) 
coord = Pyro4.Proxy(ns.lookup("BasebandCoordinator"))

coord.subscribe_uri(uri)

daemon.requestLoop()
