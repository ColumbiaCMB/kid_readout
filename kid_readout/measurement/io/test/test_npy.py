from testfixtures import TempDirectory
from kid_readout.measurement.test import utilities
from kid_readout.measurement.io import npy


def test_read_write_measurement():
    with TempDirectory() as directory:
        io = npy.IO(directory.path)
        original = utilities.get_measurement()
        name = 'measurement'
        io.write(original, name)
        assert original == io.read(name)


def test_read_write_stream():
    with TempDirectory() as directory:
        io = npy.IO(directory.path)
        original = utilities.make_stream()
        name = 'stream'
        io.write(original, name)
        assert original == io.read(name)


def test_read_write_streamarray():
    with TempDirectory() as directory:
        io = npy.IO(directory.path)
        original = utilities.make_stream_array()
        name = 'stream_array'
        io.write(original, name)
        assert original == io.read(name)
