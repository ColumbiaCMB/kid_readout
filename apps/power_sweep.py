import numpy as np

from kid_readout.utils import roach_interface, sweeps, data_file, data_block

ri = roach_interface.RoachBaseband()
df = data_file.DataFile()

ri.set_dac_attenuator(31.5)

f0s = np.load('/home/gjones/workspace/apps/first_pass_f0s_2013-11-20.npy')[:16]
f0s.sort()
#ri.set_tone_freqs(center_freqs, nsamp=2**20)
ri._sync()

df.log_hw_state(ri)
df.log_adc_snap(ri)


base_atten = 57.0 #corresponds to -80 dBm/tone for 12 tones

atten_offsets = [20]#[20,24,30,36,39,42,45]

try:
    for m in range(20):
        for k,attoff in enumerate(atten_offsets):
            ri.set_dac_attenuator(base_atten-attoff)
            df.log_hw_state(ri)
            print "starting sweep",k, "atten at", (base_atten-attoff)
            
            swp = sweeps.segmented_fine_sweep(ri, center_freqs=f0s, segments=sweeps.default_segments_mhz,nchan_per_step = 8)
            df.add_sweep(swp)
            df.log_hw_state(ri)
            df.log_adc_snap(ri)
except KeyboardInterrupt:
    pass
df.close()
    
