===============
Getting Started
===============

Using the CUKIDS readout consists of two basic parts: loading a waveform to synthesize with the DAC and reading data
from around the synthesized tones.

The basic interface to the ROACH takes place using the classes in the :mod:`kid_readout.utils.roach_interface` module.
Specifically, the :class:`kid_readout.utils.roach_interface.RoachBaseband` class is the most well maintained at present.

The class takes care of programming the ROACH (in the initialization method). It also assumes that it has access to
Valon synthesizers that provide the ADC and DAC clock signals. If this is not the case for your system,
you may need to alter the code to avoid these calls.

Waveform Synthesis
==================

The waveforms are synthesized with a continuous playback ring buffer. The length of the buffer can be set to a power
of two samples. The length directly impacts the frequency resolution of the resulting tones. The waveform synthesis
code enforces that the sinusoids in the buffer have an integer number of periods, to avoid glitches.

For example, if the buffer length is $2^16$ samples, the lowest frequency that can be generated is a tone with a
period equal to $2^16$ samples. The next highest frequency is twice that (two cycles per $2^16$ samples), and so on.

The basic function for defining and loading a waveform is
:func:`kid_readout.utils.roach_interface.RoachBaseband.set_tone_freqs` which takes an array of frequencies in MHz and
 the length of the buffer (nsamp) and computes the required waveform and loads it into the ROACH.

Note about loading the waveforms
--------------------------------

The waveforms are typically rather large (several MB), so while they can be transferred to the DRAM on the ROACH over
katcp, the overhead involved usually makes this unattractive. Right now, this option is in fact hard coded out,
in that the underlying :func:`kid_readout.utils.roach_interface.RoachBaseband.load_waveform` method has the default
value of fast=True, meaning do not use katcp for loading the DRAM. You can change this if you want to use katcp for
simplicity in spite of the overhead.

By default, the system assumes that the ROACH mounts its filesystem from the local machine at /srv/roach_boot/etch
and it will try to write a copy of the waveform to /srv/roach_boot/etch/boffiles/dram.bin. It will then use ssh to
execute a "dd" unix command to write the dram.bin file to the DRAM using BORPH. This is siginificantly faster. NFS
takes care of transferring dram.bin over the network seamlessly.

Getting data off the ROACH
==========================

The simplest method to read data from the ROACH is to use katcp, but this also will limit the number of channels you
can simultaneously stream data from. The function to use is get_data_katcp.

For faster data transfer, there is an executable that runs on the ROACH PowerPC (PPC) that reads data from the FPGA
and streams it over UDP. To use this, you need to copy the ppc source folder to the ROACH,
probably to a subdirectory of the /boffiles directory since that's often the only part of the file system that's
writable. To build, just do "gcc -o kid_ppc kid_ppc.c".

When programming the ROACH, the RoachBasband class initialization will try to start kid_ppc by default (controlled by
 the start_udp option of the initialize method).
