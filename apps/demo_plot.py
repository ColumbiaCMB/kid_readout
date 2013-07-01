from matplotlib import pyplot as plt
import numpy as np
import Pyro4
import Queue
import threading

class SimpleViewer():
    def __init__(self):
        self.fig = plt.figure()
        self.fig.show()
        self.ax = self.fig.add_subplot(111)
        self.line = None
        self.queue = Queue.Queue()
        
    def handle(self,data):
        print "got data"
        if data['type'] == 'power spectrum':
            y = np.log10(data['data'])
            x = np.arange(len(y))
            self.queue.put((x,y))

    def plot_loop(self):
        x,y = self.queue.get()
        print "plotting data"
        if self.line:
            self.line.set_data(x,y)
        else:
            self.line, = self.ax.plot(x,y)
        self.fig.canvas.draw()
        
        
ns = Pyro4.naming.locateNS()

viewer = SimpleViewer()
daemon = Pyro4.Daemon()
uri = daemon.register(viewer) 
coord = Pyro4.Proxy(ns.lookup("BasebandCoordinator"))

coord.subscribe_uri(uri,['power spectrum'])

pyro_thread = threading.Thread(target=daemon.requestLoop)
pyro_thread.daemon = True
pyro_thread.start()

while True:
    viewer.plot_loop()
