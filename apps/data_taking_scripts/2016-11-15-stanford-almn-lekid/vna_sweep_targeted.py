from kid_readout.interactive import *

#setup = hardware.Hardware()

ri = Roach2Baseband()

ri.set_modulation_output('high')

frequencies = np.array([112.648,
                        117.832,
                        144.524,
                        145.231,
                        154.524,
                        158.278,
                        164.012, #null
                        167.111, # null
                        ])

for dac_atten in [60,50,40,30,20]:
    ri.set_dac_atten(dac_atten)

    df = acquire.new_nc_file(suffix='vna_dac_atten_%.1f_dB' % dac_atten)
    swa = acquire.run_sweep(ri,frequencies[None,:]+np.arange(-240,240,dtype='int')[:,None]*512./2.**19,
                            2**19,
                            verbose=True,length_seconds=.25,
                            )
#    swa = acquire.run_sweep(ri,np.linspace(10,250,256)[None,:]+np.arange(480,dtype='int')[:,None]*512./2.**18,
#                            2**18,
#                            verbose=True,length_seconds=.25,
#                            )
    df.write(swa)
    df.close()