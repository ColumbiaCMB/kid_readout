import time
def stabilityTest():
    while True:
        tic = time.time()
        try:
            d,addrs = spr.getData(16)
            s0,s1 = spr.getRawAdc()
        except Exception,e:
            print e
            continue
        tstr = time.strftime('%Y-%m-%d-%H%M%S')
        fname = '/home/gjones/data/%s' % tstr
        np.savez(fname,d=d,addrs=addrs,tic=tic,s0=s0,s1=s1)
        print fname
        time.sleep(60)