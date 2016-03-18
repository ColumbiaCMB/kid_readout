"""
Basic framework for acquiring a roach measurement that includes both sweep(s) and stream(s).

Acquire
-Initialize equipment.
-Initialize roach: preload frequencies, if necessary.
-Create state dictionary containing state from all equipment, including temperatures, if possible.
-Run a coarse sweep, if necessary: create a SweepArray and extract resonance frequencies.
-Run fine sweeps to map out resonance frequencies carefully.
If desired, we can combine the data from coarse and fine sweeps into a single SweepArray.
All streams in these sweeps are created with the same roach state, which should not change during the sweeps.
The sweep(s) are created with the experiment state, which should also not change.

Acquire streams:
-Initialize equipment for stream(s).
-Initialize roach for stream(s).
-Create experiment state dictionary.
-Acquire a StreamArray.
-Repeat the stream acquisition as needed
-Instantiate the final measurement with all data, and save it to disk.
-Clean up equipment.

If instead we want to save data as it is collected, we can do that by writing a blank final measurement to disk, then
writing the sub-measurements as they are acquired.
"""
from __future__ import division
import numpy as np
from kid_readout.measurement import multiple


def load_sweep_tones(ri, tone_banks, num_tone_samples):
    ri.set_tone_freqs(np.vstack(tone_banks), nsamp=num_tone_samples)


def run_sweep(ri, tone_banks, num_tone_samples, length_seconds=1, tones_loaded=False, state=None, description=''):
    stream_arrays = []
    for n, tone_bank in enumerate(tone_banks):
        if tones_loaded:
            ri.select_bank(n)
        else:
            ri.set_tone_freqs(tone_bank, nsamp=num_tone_samples)
        ri.select_fft_bins(np.arange(tone_bank.size))
        stream_arrays.append(ri.get_measurement(num_seconds=length_seconds))
    return multiple.SweepArray(stream_arrays, state=state, description=description)
