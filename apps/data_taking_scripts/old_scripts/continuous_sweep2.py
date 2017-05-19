import numpy as np

from kid_readout.roach import baseband
from kid_readout.utils import sweeps, data_file, data_block


ri = baseband.RoachBaseband()
df = data_file.DataFile()

ri.set_adc_attenuator(16)
ri.set_dac_attenuator(16)

center_freqs = np.array([92.94,
                         96.31,
                         101.546,
                         103.195,
#                         121.35,
#                         130.585,
#                         133.436,
#                         148.238,
#                         148.696,
#                         148.867,
#                         149.202,
#                         159.572,
#                         167.97,
#                         172.93,
#                         176.645,
#                         178.764
])


ri.set_tone_freqs(center_freqs, nsamp=2**20)
ri._sync()

df.log_hw_state(ri)
df.log_adc_snap(ri)


def fine_sweep(ri,center_freqs, sweep_width = 0.1,npoints =128,nsamp=2**20):
    offsets = np.linspace(-sweep_width/2.0, sweep_width/2.0, npoints)
    swp = data_block.SweepData()
    
    def callback(block):
        swp.add_block(block)
        return False
    
    for k,offset in enumerate(offsets):
        print "subsweep",k,"of",npoints
        sweeps.coarse_sweep(ri, center_freqs + offset, nsamp=nsamp, nchan_per_step=4, reads_per_step=2, callback=callback,sweep_id=k)
    return swp

nsweeps = 0
while True:
    try:
        print "starting sweep",nsweeps
        df.log_hw_state(ri)
        df.log_adc_snap(ri)
        swp = fine_sweep(ri,center_freqs)
        df.add_sweep(swp)
        nsweeps += 1
    except KeyboardInterrupt:
        df.close()
        break
        
    
