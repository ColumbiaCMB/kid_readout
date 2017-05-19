__author__ = 'gjones'
from matplotlib import pyplot as plt
import time
plt.ion()

plt.plot(range(10))
plt.show()

print "waiting"
time.sleep(10)


print "plotting"
plt.plot(range(10)[::-1])

print "waiting"
time.sleep(10)

print "showing"
plt.show()