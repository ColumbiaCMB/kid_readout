# kid_readout

Code for the ROACH CUKIDS readout and analysis.

[![Build Status](https://travis-ci.org/ColumbiaCMB/kid_readout.svg?branch=master)](https://travis-ci.org/ColumbiaCMB/kid_readout)

## Disclaimer

This code is provided in the hope that it may be useful to others, but is still very much a work in progress.


## FPGA designs

Using the ROACH code requires the FPGA designs available here:
* Simulink .slx files (public): https://github.com/ColumbiaCMB/kid_readout_fpga
* Compiled .bof files (private): https://github.com/ColumbiaCMB/kid_readout_fpga_bof


## Structure
* `apps/` Scripts, mostly for acquiring and analyzing data
* `docs/` Documentation
* `kid_readout/` All library code
  * `interactive.py` Import * from this for interactive use
  * `analysis/` Resonator fitting, timeseries analysis, experiment info (in resources/)
  * `equipment/` Scripts and library code for lab equipment
  * `measurement/` Code for organizing collected data and writing it to disk
  * `roach/` Communicate with ROACH 1 and 2 boards
  * `settings/` Local variables and settings
  * `utils/` Miscellany, mostly deprecated
*  `ppc/` ROACH PPC code


## Install

From the directory where this package was cloned, first create an environment named kid_readout with all dependencies:
`conda env create -f environment.yml`

Activate the kid_readout environment:
`$ source activate kid_readout`

At this point, all of the non-hardware tests should pass:
`$ nosetests -v`

All dependencies should be installed, so it should be possible to install the package in development mode:
`$ pip install --no-deps -e .`

This is a hack to stop the `corr` package from importing lots of things we don't use, one of which raises an error:
`$ echo "" > $CONDA_PREFIX/lib/python2.7/site-packages/corr/__init__.py`
At this point, all the hardware tests should pass if the corresponding readout hardware is connected:
```
$ cd kid_readout/roach/tests
$ nosetests -v test_roach2_heterodyne_loopback.py
```
