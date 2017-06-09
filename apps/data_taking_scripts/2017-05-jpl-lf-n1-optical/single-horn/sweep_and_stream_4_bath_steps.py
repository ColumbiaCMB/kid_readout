from kid_readout.interactive import *
import time

#setup = hardware.Hardware()

ri = Roach2Baseband()

ri.set_modulation_output('high')
initial_f0s = np.load('/data/readout/resonances/2017-06-JPL-8x8-LF-N1_single_horn_4.npy')/1e6

nf = len(initial_f0s)
atonce = 4
if nf % atonce > 0:
    print "extending list of resonators to make a multiple of ", atonce
    initial_f0s = np.concatenate((initial_f0s, np.arange(1, 1 + atonce - (nf % atonce)) + initial_f0s.max()))

nsamp = 2**20 #going above 2**18 with 128 simultaneous tones doesn't quite work yet
offsets = np.arange(-16,16)*512./nsamp

dac_atten = 35
ri.set_dac_atten(dac_atten)

cold_temp = [100,125,150,175]

for temp in cold_temp:
    raw_input('change bath temp to %3.3f' %temp)
    if temp == 100:
        print "temp is 100, skipping wait"
    else:
        print "waiting 15 minutes", temp
        time.sleep(900)

    tic = time.time()

    ncf = new_nc_file(suffix='%d_dB_dac_bath_temp_%.3f_mK' %(dac_atten, temp))
    swpa = acquire.run_sweep(ri, tone_banks=initial_f0s[None,:] + offsets[:,None], num_tone_samples=nsamp,
                                 length_seconds=0, verbose=True,
                             )
    ncf.write(swpa)
    current_f0s = []
    for sidx in range(initial_f0s.shape[0]):
        swp = swpa.sweep(sidx)
        res = swp.resonator
        print res.f_0, res.Q, res.current_result.redchi, (initial_f0s[sidx]*1e6-res.f_0)
        if np.abs(res.f_0 - initial_f0s[sidx]*1e6) > 200e3:
            current_f0s.append(initial_f0s[sidx]*1e6)
            print "using original frequency for ",initial_f0s[sidx]
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
    ri.select_fft_bins(range(initial_f0s.shape[0]))
    raw_input("turn off compressor")
    meas = ri.get_measurement(num_seconds=30., description='compressor off')
    raw_input("turn on compressor")
    ncf.write(meas)
    print "dac_atten %f done in %.1f minutes" % (dac_atten, (time.time()-tic)/60.)
    ncf.close()

ri.set_dac_atten(20)