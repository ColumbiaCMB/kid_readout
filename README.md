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
        adc.py (soon to be depreciated)
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

Testing https and ssh -daniel
