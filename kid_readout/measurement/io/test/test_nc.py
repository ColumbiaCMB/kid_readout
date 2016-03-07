import os
from testfixtures import TempDirectory
from kid_readout.measurement import core
from kid_readout.measurement.test.utilities import compare_measurements, get_measurement
from kid_readout.measurement.io import nc


def test_read_write():
    with TempDirectory() as directory:
        filename = 'test.nc'
        io = nc.IO(os.path.join(directory.path, filename))
        original = get_measurement()
        name = 'test'
        core.write(original, io, name)
        compare_measurements(original, core.read(io, name))
