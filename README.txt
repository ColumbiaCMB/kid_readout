kid_readout
===========
Code for the ROACH KID readout.

=== Libraries used  which should be installed ===

Currently running on CentOS 6.5
 Python 2.7.6

    ./configure –prefix=/home/local
    create /etc/ld.so.conf.d/python27.conf with /home/local/lib in it
    Arg forgot to make sure tcl tk headers were installed (they weren't). Use package manager to get tcl-devel and tk-devel packages
    do same ./configure –prefix=/home/local
    make install

setuptools-2.0

    python2.7 setup.py install

pip 1.4.1

    download get-pip.py
    python2.7 get-pip.py

numpy 1.8

    pip install numpy

scipy 0.13.2

    pip install scipy

matplotlib 1.3.1

    pip install matplotlib
    gave some complaint about installing an externally hosted file, who cares

ipython 1.1.0

    pip install ipython

SIP 4.15.3

    pip install SIP fails as expected because it doesn't use setup.py so….
    mkdir /home/local/dl/build # should have done this before
    pip –build=/home/local/dl/build install SIP #fails
    cd /home/local/dl/build/SIP
    python2.7 configure.py
    make install

ran into obnoxious compatibility issues with latest SIP and PyQt 4.10.3, I thought it was just picking up the wrong sip.h, but even when manually renaming that etc. it still croaked. tried a few combos until finding one that seems to work:
sip 4.14.7

    python2.7 configure.py
    make install

pyqt 4.9.6

    configure.py –qmake=/usr/lib64/qt4/bin/qmake -k
    This didn't work because it created some weird static libraries that couldn't be imported
    python2.7 configure.py –qmake=/usr/lib64/qt4/bin/qmake -g –verbose
    make install
    seems to work (ipython –pylab=qt produces a plot)

katcp 0.5

    pip install katcp

pyzmq-14.0.1

    pip install pyzmq

Jinja2-2.7.1.tar.gz

    pip install jinja2

corr (for katcp_wrapper)

    just copy corr directory to /home/local/lib/python2.7/site-packages and remove .pyc files

lmfit 0.7.2

    pip install lmfit

sphinx 1.2

    pip install sphinx

pandas 0.12

    pip install pandas

scikit-learn 0.14.1

    pip install scikit-learn

netcdf4 1.0.7

    pip install netcdf4

valon

    copy from ~readout/lib and remove .pyc files

pyserial 2.7

    pip install pyserial

spyder 2.2.5

    pip install spyder

sympy 0.7.4

    pip install sympy



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
