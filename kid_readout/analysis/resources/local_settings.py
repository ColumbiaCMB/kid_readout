import os
import socket

hostname = socket.gethostname()
BASE_DATA_DIR = os.path.join('/data',hostname)
if not os.path.exists(BASE_DATA_DIR):
    print "no data directory set up for",hostname,"defaulting to /data"
    BASE_DATA_DIR = '/data'

