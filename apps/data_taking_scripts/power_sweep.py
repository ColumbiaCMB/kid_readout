import numpy as np

from kid_readout.roach import baseband
from kid_readout.utils import sweeps, data_file


default_segments_hz = [#np.arange(0,200e3,8e3)-490e3,
                       #np.arange(200e3,360e3,4e3)-490e3,
                       #np.arange(360e3,440e3,2e3)-490e3,
#                                  np.arange(440e3,480e3,1e3)-490e3,
                                  np.arange(480e3,510e3,0.5e3)-490e3]

default_segments_mhz = [x/1e6 for x in default_segments_hz]

ri = baseband.RoachBaseband()
#ri.initialize()
df = data_file.DataFile()

#f0s = np.load('/home/gjones/workspace/apps/first_pass_sc3x3_0813f9.npy')
f0s = np.load('/home/gjones/workspace/apps/sc5x4_0813f10_first_pass.npy')

ri.set_tone_freqs(f0s, nsamp=2**20)
f0s.sort()
#ri.set_tone_freqs(center_freqs, nsamp=2**20)
ri._sync()

#df.log_hw_state(ri)
df.log_adc_snap(ri)


base_atten = 50.0 #corresponds to -80 dBm/tone at readout  (-120 dBm at device) for 8 tones

atten_offsets = [0,10,15,20,25,30]

try:
    for m in range(1):
        for k,attoff in enumerate(atten_offsets):
            ri.set_dac_attenuator(base_atten-attoff)
            df.log_hw_state(ri)
            print "starting sweep",k, "atten at", (base_atten-attoff)
            
            swp = sweeps.segmented_fine_sweep(ri, center_freqs=f0s, segments=default_segments_mhz,nchan_per_step = 8,reads_per_step=16)
            df.add_sweep(swp)
            df.log_hw_state(ri)
            df.log_adc_snap(ri)
except KeyboardInterrupt:
    pass
df.close()
    
