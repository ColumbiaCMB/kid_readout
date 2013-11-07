kid_readout
===========
Code for the ROACH KID readout.

=== Structure ===
As of 2013-11-07, the project structure is as follows:
repository_base/
  kid_readout/
    __init__.py
    utils/
      __init__.py
      <almost everything lives here>
    analysis/
      __init__.py
      <fitting libraries live here>
    apps/
      gain_phase_characterization
      <scripts and non-library code>
    ppc/
      <Bjorn's timing code?>


=== Install ===
To install the kid_readout package, you need to add the directory in
which you cloned this repository to your PYTHONPATH environment
variable.  To be more specific, if the repository lives in
/home/user/kid_readout.git,
add this line to your .bash_profile (or .bashrc) file:
export PYTHONPATH=$PYTHONPATH:/home/user/kid_readout.git
Then, statements like
from kid_readout.utils import whatever
should work.


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
