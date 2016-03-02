import pandas as pd
from kid_readout.measurement import core


def test_measurement_instantiation():
    m = core.Measurement()
    assert m.state == {}


def test_measurement_add_origin():
    m = core.Measurement()
    df = m.to_dataframe()
    assert df is None
    df = pd.DataFrame([0])  # This creates a DataFrame with shape (1, 1).
    m.add_origin(df)
    assert df.shape == (1, 1 + 3)
    assert df.io_class.iloc[0] is None
    assert df.root_path.iloc[0] is None
    assert df.node_path.iloc[0] is None


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
        except core.MeasurementError:
            assert True
    for good in good_node_paths:
        try:
            core.validate_node_path(good)
            assert True
        except core.MeasurementError:
            assert False


