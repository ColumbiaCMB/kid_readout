from testfixtures import TempDirectory
from kid_readout.measurement.test import utilities
from kid_readout.measurement.io import npy


def test_read_write_measurement():
    with TempDirectory() as directory:
        io = npy.NumpyDirectory(directory.path)
        original = utilities.CornerCases()
        name = 'measurement'
        io.write(original, name)
        assert original == io.read(name)


def test_read_write_stream():
    with TempDirectory() as directory:
        io = npy.NumpyDirectory(directory.path)
        original = utilities.fake_single_stream()
        name = 'stream'
        io.write(original, name)
        assert original == io.read(name)


def test_read_write_streamarray():
    with TempDirectory() as directory:
        io = npy.NumpyDirectory(directory.path)
        original = utilities.fake_stream_array()
        name = 'stream_array'
        io.write(original, name)
        assert original == io.read(name)


def test_read_write_sweeparray():
    with TempDirectory() as directory:
        io = npy.NumpyDirectory(directory.path)
        original = utilities.fake_sweep_array()
        name = 'sweep_array'
        io.write(original, name)
        assert original == io.read(name)


def test_read_write_sweepstreamarray():
    with TempDirectory() as directory:
        io = npy.NumpyDirectory(directory.path)
        original = utilities.fake_sweep_stream_array()
        name = 'sweep_stream_array'
        io.write(original, name)
        assert original == io.read(name)


def test_memmap():
    with TempDirectory() as directory:
        io = npy.NumpyDirectory(directory.path, memmap=True)
        original = utilities.fake_single_stream()
        name = 'stream'
        io.write(original, name)
        assert original == io.read(name)
