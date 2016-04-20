import os
import numpy as np
from testfixtures import TempDirectory

from kid_readout.measurement.test import utilities
from kid_readout.measurement.io import nc


def test_read_write_measurement():
    with TempDirectory() as directory:
        filename = 'test.nc'
        io = nc.NCFile(os.path.join(directory.path, filename))
        original = utilities.CornerMeasurement()
        name = 'measurement'
        io.write(original, name)
        assert original == io.read(name)


def test_read_write_stream():
    with TempDirectory() as directory:
        filename = 'test.nc'
        io = nc.NCFile(os.path.join(directory.path, filename))
        original = utilities.make_stream()
        name = 'stream'
        io.write(original, name)
        assert original == io.read(name)


def test_read_write_streamarray():
    with TempDirectory() as directory:
        filename = 'test.nc'
        io = nc.NCFile(os.path.join(directory.path, filename))
        original = utilities.make_stream_array()
        name = 'stream_array'
        io.write(original, name)
        assert original == io.read(name)


def test_cached_single_stream():
    with TempDirectory() as directory:
        filename = 'test.nc'
        io = nc.NCFile(os.path.join(directory.path, filename), cache_s21_raw=True)
        original = utilities.make_stream()
        name = 'measurement'
        io.write(original, name)
        assert np.all(original.s21_raw == io.read(name).s21_raw)


def test_cached_stream_array():
    with TempDirectory() as directory:
        filename = 'test.nc'
        io = nc.NCFile(os.path.join(directory.path, filename), cache_s21_raw=True)
        original = utilities.make_stream_array()
        name = 'measurement'
        io.write(original, name)
        assert np.all(original.s21_raw == io.read(name).s21_raw)



