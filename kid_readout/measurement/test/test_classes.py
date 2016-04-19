"""
Tests for the version module.

Note that several of the tests modify the dictionaries in version.py and reset them at the end of the test. Otherwise,
the changes would persist between tests.
"""
from copy import deepcopy

from kid_readout.measurement import core, classes
from kid_readout.measurement.io import memory


class MovedMeasurement(core.Node):

    def __init__(self, state=None, description=''):
        pass


class OldUnversioned(core.Measurement):

    # This version info will be saved, but None is the value used for old-style classes.
    _version = None

    dimensions = {}

    # Save this class with the old-style fully-qualified class name.
    @classmethod
    def class_name(cls):
        return cls.__module__ + '.' + cls.__name__


class NewVersioned(core.Measurement):

    _version = 0

    dimensions = {}


def test_all_versions_exist():
    for class_name, version_dict in classes._versioned.items():
        for version_number, full_class_name in version_dict.items():
            core.get_class(full_class_name)
    for original_full_class_name, full_class_name in classes._unversioned.items():
        core.get_class(full_class_name)


def test_add_new_version():
    io = memory.Dictionary(None)
    from_core = core.Measurement()
    assert from_core._version == 0
    from_core._version = 1
    name = 'from_core'
    io.write(from_core, name)
    _versioned = deepcopy(classes._versioned)
    # Add a new entry for version 1:
    classes._versioned['Measurement'][1] = __package__ + '.test_classes.MovedMeasurement'
    try:
        from_here = io.read(name)
        assert from_here.__class__.__module__ == __package__ + '.test_classes'
    finally:
        classes._versioned = _versioned


def test_change_class_module():
    io = memory.Dictionary(None)
    from_core = core.Measurement()
    assert from_core._version == 0
    name = 'from_core'
    io.write(from_core, name)
    _versioned = deepcopy(classes._versioned)
    # Change the location of Measurement version 0:
    classes._versioned['Measurement'][0] = __package__ + '.test_classes.MovedMeasurement'
    try:
        from_here = io.read(name)
        assert from_here.__class__.__module__ == __package__ + '.test_classes'
    finally:
        classes._versioned = _versioned


def test_unversioned():
    io = memory.Dictionary(None)
    old = OldUnversioned()
    name = 'old'
    io.write(old, name)
    assert io.read(name) == old
    _unversioned = deepcopy(classes._unversioned)
    classes._unversioned[__package__ + '.test_classes.OldUnversioned'] = __package__ + '.test_classes.NewVersioned'
    try:
        new = io.read(name)
        assert new.class_name() == NewVersioned.__name__
    finally:
        classes._unversioned = _unversioned
