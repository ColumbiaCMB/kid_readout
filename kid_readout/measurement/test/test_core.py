import numpy as np
import pandas as pd
from kid_readout.measurement import core
from kid_readout.measurement.io import memory
from kid_readout.measurement.test.utilities import get_measurement


def test_measurement_instantiation():
    m = core.Measurement({})
    assert m.state == core.StateDict()
    assert m.description == 'Measurement'
    assert m._parent is None
    assert m._io_class is None
    assert m._root_path is None
    assert m._node_path is None
    try:
        m.analyze()
    except:
        assert False


def test_measurement_to_dataframe():
    assert all(core.Measurement({}).to_dataframe() == pd.DataFrame())


def test_measurement_add_origin():
    m = core.Measurement({})
    df = m.to_dataframe()
    assert df is None
    df = pd.DataFrame([0])  # This creates a DataFrame with shape (1, 1).
    m.add_origin(df)
    assert df.shape == (1, 1 + 3)
    s = df.iloc[0]
    assert s.io_class is None
    assert s.root_path is None
    assert s.node_path is None


def test_measurement_add_legacy_origin():
    m = core.Measurement({})
    df = m.to_dataframe()
    assert df is None
    df = pd.DataFrame([0])  # This creates a DataFrame with shape (1, 1).
    m.add_legacy_origin(df)
    assert df.shape == (1, 1 + 2)
    s = df.iloc[0]
    assert s.io_module == 'kid_readout.measurement.legacy'
    assert s.root_path is None


def test_measurement_sequence():
    length = int(100 * np.random.random())
    contents = np.random.random(length)
    mt = core.instantiate_sequence('kid_readout.measurement.core.MeasurementTuple', contents)
    assert np.all(mt == contents)
    assert core.is_sequence(mt.__module__ + '.' + mt.__class__.__name__)
    assert mt.shape == (length,)
    ml = core.instantiate_sequence('kid_readout.measurement.core.MeasurementList', contents)
    assert np.all(ml == contents)
    assert core.is_sequence(mt.__module__ + '.' + ml.__class__.__name__)
    assert ml.shape == (length,)


def test_read_write():
    io = memory.IO(None)
    original = get_measurement()
    name = 'test'
    core.write(original, io, name)
    assert original == core.read(io, name)


def test_comparison_code_state():
    m1 = get_measurement()
    m2 = get_measurement()
    m1.state['test'] = 1
    m2.state['test'] = 2
    assert m1 != m2


def test_comparison_code_attribute():
    m1 = get_measurement()
    m2 = get_measurement()
    m1.attribute = 1
    m2.attribute = 2
    assert m1 != m2


def test_instantiate():
    full_class_name = 'kid_readout.measurement.core.Measurement'
    variables = {'state': {'key': 'value'},
                 'description': 'instantiated Measurement',
                 'extra_variable': [0]}
    m = core.instantiate(full_class_name, variables, extras=True)
    assert m.state == core.StateDict(variables['state'])
    assert m.description == variables['description']
    assert m.extra_variable == variables['extra_variable']


def test_join():
    nodes = ['one', 'two', 'three']
    assert core.join(nodes[0]) == 'one'
    assert core.join(*nodes) == 'one:two:three'


def test_split():
    assert core.split('') == ''
    assert core.split('one') == ('', 'one')
    assert core.split('one:two:three') == ('one:two', 'three')


def test_explode():
    assert core.explode('boom') == ['boom']
    assert core.explode('boom:kaboom') == ['boom', 'kaboom']


def test_validate_node_path():
    bad_node_paths = ['', ' ', '"', ':', '/', '\\', '?', 'node:', ':node', 'node:path:', 'bad-hyphen', 'exclamation!']
    good_node_paths = ['good', 'very:good', 'EXTREMELY:GOOD', 'underscore_is_fine:_:__really__', '0:12:345']
    for bad in bad_node_paths:
        try:
            core.validate_node_path(bad)
            assert False
        except core.MeasurementError:
            pass
    for good in good_node_paths:
        try:
            core.validate_node_path(good)
        except core.MeasurementError:
            assert False
