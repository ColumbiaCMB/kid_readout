from testfixtures import TempDirectory
from kid_readout.measurement import core
from kid_readout.measurement.test.utilties import compare_measurements, get_measurement
from kid_readout.measurement.io import npy


def test_read_write():
    try:
        directory = TempDirectory()
        io = npy.IO(directory.path)
        original = get_measurement()
        name = 'test'
        core.write(original, io, name, close=False)
        compare_measurements(original, core.read(io, name))
    finally:
        directory.cleanup()