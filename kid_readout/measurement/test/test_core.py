import copy
import numpy as np
import pandas as pd
from kid_readout.measurement import core, basic
from kid_readout.measurement.io import memory
from kid_readout.measurement.test import utilities


def test_measurement_instantiation_blank():
    m = core.Measurement()
    assert m.state == core.StateDict()
    assert m.description == ''
    assert m._parent is None
    assert m._io_class is None
    assert m._root_path is None
    assert m._node_path is None


def test_measurement_to_dataframe():
    assert all(core.Measurement().to_dataframe() == pd.DataFrame())


def test_measurement_add_origin():
    m = core.Measurement()
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
    m = core.Measurement()
    df = m.to_dataframe()
    assert df is None
    df = pd.DataFrame([0])  # This creates a DataFrame with shape (1, 1).
    m.add_legacy_origin(df)
    assert df.shape == (1, 1 + 2)
    s = df.iloc[0]
    assert s.io_module == 'kid_readout.measurement.legacy'
    assert s.root_path is None


def test_measurement_list():
    length = 3
    contents = [utilities.CornerCases() for n in range(length)]
    ml = core.instantiate_sequence('kid_readout.measurement.core.MeasurementList', contents)
    assert np.all(ml == contents)
    assert core.is_sequence(ml.__module__ + '.' + ml.__class__.__name__)
    assert len(ml) == length


def test_io_list():
    num_streams = 3
    streams = core.MeasurementList([utilities.CornerCases() for n in range(num_streams)])
    io = memory.Dictionary()
    sweep = basic.SingleSweep(core.IOList())
    io.write(sweep)
    sweep.streams.extend(streams)
    assert io.read(io.measurement_names()[0]) == basic.SingleSweep(streams)


def test_read_write():
    io = memory.Dictionary()
    original = utilities.CornerCases()
    name = 'test'
    io.write(original, name)
    assert original == io.read(name)


def test_eq_state():
    m1 = utilities.CornerCases()
    m2 = utilities.CornerCases()
    assert m1 == m2
    m1.state['test'] = 1
    m2.state['test'] = 2
    assert m1 != m2


def test_eq_array():
    m1 = utilities.fake_single_stream()
    m2 = basic.SingleStream(**dict([(k, copy.copy(v)) for k, v in m1.__dict__.items() if not k.startswith('_')]))
    assert m1 == m2
    index = np.random.random_integers(0, m1.s21_raw.size)
    m2.s21_raw[index] += 1
    assert m1 != m2
    m1.s21_raw[index] = m2.s21_raw[index] = np.nan
    assert m1 == m2


def test_comparison_code_attribute():
    m1 = utilities.CornerCases()
    m2 = utilities.CornerCases()
    m1.attribute = 1
    m2.attribute = 2
    assert m1 != m2


def test_instantiate():
    full_class_name = 'kid_readout.measurement.core.Measurement'
    variables = {'state': {'key': 'value'},
                 'description': 'instantiated Measurement'}
    m = core.instantiate(full_class_name, variables)
    assert m.state == core.StateDict(variables['state'])
    assert m.description == variables['description']


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
    bad_node_paths = ['', ' ', '"', ':', '/', '\\', '?', '!', 'node:', ':node', 'node:path:', 'bad-hyphen', '0number']
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


def test_sweep_stream_array_node_path():
    ssa = utilities.fake_sweep_stream_array()
    assert ssa.node_path == ''
    assert ssa.sweep_array.node_path == 'sweep_array'
    assert ssa.sweep_array.stream_arrays.node_path == 'sweep_array:stream_arrays'
    assert ssa.sweep_array.stream_arrays[0].node_path == 'sweep_array:stream_arrays:0'
    io = memory.Dictionary()
    name = 'ssa'
    io.write(ssa, name)
    assert ssa.node_path == 'ssa'
    assert ssa.sweep_array.node_path == 'ssa:sweep_array'
    assert ssa.sweep_array.stream_arrays.node_path == 'ssa:sweep_array:stream_arrays'
    assert ssa.sweep_array.stream_arrays[0].node_path == 'ssa:sweep_array:stream_arrays:0'

