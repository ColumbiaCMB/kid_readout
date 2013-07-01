kid_readout
===========
Code for the ROACH KID readout.

=== structure ===
The project structure is as follows:
kid_readout/
    apps/
        gain_phase_characterization
        ...
    utils/
        single_pixel.py
    output/

=== install ===
To install the kid_readout package, you need to add the directory in which you cloned
kid_readout to your PYTHONPATH environment variable, by adding this line to 
your .bashrc file:
export $PYTHONPATH:/path/to/kid
To be more specific:
if kid_readout lives in /home/user/git/kid_readout, 
you should add the following line to .bashrc: 
export $PYTHONPATH:/home/user/git
notice that kid_readout is NOT in the directory.


=== Proposed class structure ===

(SinglePixel)Readout - abstract interface to FPGA readout hardware
	provides functions to setup output waveforms and select FFT bins to read out, and eventually synchronization

Coordinator - class to tie everything together and maintain system state. 
	Inherits from (SinglePixel)(Baseband) to provide FPGA access and system state
	Provides functions such as:
		start_recording - set up a data file and start recording data to it
		stop_recording - 
		subscribe - add subscription for data products
	Has attributes of:
		catcher - gets data from FPGA, either with Katcp or UDP
		writer - a disk writing class
		aggregator - processes the raw data and creates higher order data products. sends full rate data to writer.
			sends higher order products to any subscribers
		
A viewer or controller application can then grab a Pyro proxy of the Coordinator, subscribe to the desired 
data products and then use functions to configure and start recording.
