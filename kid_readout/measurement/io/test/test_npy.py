from testfixtures import TempDirectory
from kid_readout.measurement import core
from kid_readout.measurement.test.utilities import compare_measurements, get_measurement
from kid_readout.measurement.io import npy


def test_read_write():
    with TempDirectory() as directory:
        io = npy.IO(directory.path)
        original = get_measurement()
        name = 'test'
        core.write(original, io, name)
        assert original == core.read(io, name)
