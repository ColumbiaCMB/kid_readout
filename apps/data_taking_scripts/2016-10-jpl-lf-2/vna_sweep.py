from kid_readout.interactive import *

#setup = hardware.Hardware()

ri = Roach2Baseband()



for dac_atten in [40]:
    ri.set_dac_atten(dac_atten)

    df = acquire.new_nc_file(suffix='vna_dac_atten_%.1f_dB' % dac_atten)
    swa = acquire.run_sweep(ri,np.linspace(110,170,128)[None,:]+np.arange(480,dtype='int')[:,None]*512./2.**18,
                            2**18,
                            verbose=True,length_seconds=.5,
                            )
    df.write(swa)
    df.close()