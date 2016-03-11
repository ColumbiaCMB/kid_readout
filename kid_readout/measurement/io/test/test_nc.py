import os
from testfixtures import TempDirectory
from kid_readout.measurement import core, single, multiple
from kid_readout.measurement.test.utilities import get_measurement
from kid_readout.measurement.io import nc


def test_read_write_measurement():
    with TempDirectory() as directory:
        filename = 'test.nc'
        io = nc.IO(os.path.join(directory.path, filename))
        original = get_measurement()
        name = 'measurement'
        core.write(original, io, name)
        assert original == core.read(io, name)


# temporarily broken
"""
def test_read_write_stream():
    with TempDirectory() as directory:
        filename = 'test.nc'
        io = nc.IO(os.path.join(directory.path, filename))
        original = single.make_stream()
        name = 'stream'
        core.write(original, io, name)
        assert original == core.read(io, name)


def test_read_write_streamarray():
    with TempDirectory() as directory:
        filename = 'test.nc'
        io = nc.IO(os.path.join(directory.path, filename))
        original = multiple.make_stream_array()
        name = 'stream_array'
        core.write(original, io, name)
        assert original == core.read(io, name)
"""