from kid_readout.interactive import *

#setup = hardware.Hardware()

ri = Roach2Baseband()

ri.set_modulation_output('high')

for dac_atten in [10]:
    ri.set_dac_atten(dac_atten)

    fstart = 3200
    fstop = 3600

    df = acquire.new_nc_file(suffix='vna_dac_atten_%.1f_dB' % dac_atten)
    swa = acquire.run_sweep(ri,np.linspace(fstart,fstop,128)[None,:]+np.arange(480,dtype='int')[:,None]*512./2.**18,
                            2**18,
                            verbose=True,length_seconds=.1,
                            )
    df.write(swa)
    df.close()


#for example
#170-230 MHz band, steps are (230-170)/128
#then sampling 480 times between each of these steps by stepping an additional 2**18