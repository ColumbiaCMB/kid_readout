from kid_readout.interactive import *
import time

#setup = hardware.Hardware()

ri = Roach2Baseband()
ri.set_fft_gain(6)

ri.set_modulation_output('high')
initial_f0s = np.load('/data/readout/resonances/2016-11-16-stanford-almn-6-resonators.npy')

initial_f0s = initial_f0s*(1+200e-6)
nf = len(initial_f0s)
atonce = 8
if nf % atonce > 0:
    print "extending list of resonators to make a multiple of ", atonce
    initial_f0s = np.concatenate((initial_f0s, np.arange(1, 1 + atonce - (nf % atonce)) + initial_f0s.max()))

nsamp = 2**20 #going above 2**18 with 128 simultaneous tones doesn't quite work yet
offsets = np.arange(-64,64)*512./nsamp

last_f0s = initial_f0s
ncycle = 0
while True:
    print "cycle",ncycle
    print " "
    for dac_atten in [39,36]:
        tic = time.time()
        ri.set_dac_atten(dac_atten)
        ncf = new_nc_file(suffix='%d_dB_dac' % dac_atten)
        swpa = acquire.run_sweep(ri, tone_banks=last_f0s[None,:] + offsets[:,None], num_tone_samples=nsamp,
                                     length_seconds=0, verbose=True,
                                 )
        ncf.write(swpa)
        current_f0s = []
        for sidx in range(last_f0s.shape[0]):
            swp = swpa.sweep(sidx)
            res = swp.resonator
            print res.f_0, res.Q, res.current_result.redchi, (last_f0s[sidx]*1e6-res.f_0)
            if np.abs(res.f_0 - last_f0s[sidx]*1e6) > 200e3:
                current_f0s.append(last_f0s[sidx]*1e6)
                print "using original frequency for ",last_f0s[sidx]
            else:
                current_f0s.append(res.f_0)
        print "fits complete", (time.time()-tic)/60.
        current_f0s = np.array(current_f0s)/1e6
        current_f0s.sort()
        if np.any(np.diff(current_f0s)<0.015):
            print "problematic resonator collision:",current_f0s
            print "deltas:",np.diff(current_f0s)
            problems = np.flatnonzero(np.diff(current_f0s)<0.015)+1
            current_f0s[problems] = (current_f0s[problems-1] + current_f0s[problems+1])/2.0
        if np.any(np.diff(current_f0s)<0.015):
            print "repeated problematic resonator collision:",current_f0s
            print "deltas:",np.diff(current_f0s)
            problems = np.flatnonzero(np.diff(current_f0s)<0.015)+1
            current_f0s[problems] = (current_f0s[problems-1] + current_f0s[problems+1])/2.0
        ri.set_tone_freqs(current_f0s,nsamp)
        ri.select_fft_bins(range(last_f0s.shape[0]))
        last_f0s = current_f0s
        #raw_input("turn off compressor")
        meas = ri.get_measurement(num_seconds=30., description='source off stream')
        ncf.write(meas)
        print "dac_atten %f done in %.1f minutes" % (dac_atten, (time.time()-tic)/60.)
        ncf.close()
    ncycle += 1

ri.set_dac_atten(40)