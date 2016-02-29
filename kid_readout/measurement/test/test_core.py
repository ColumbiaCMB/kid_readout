from kid_readout.measurement import core


def test_measurement_instantiation():
    m = core.Measurement()
    assert m.state == {}


def test_measurement_to_dataframe():
    m = core.Measurement()
    data = {'frequency': 1.2}
    initial_entries = len(data)
    added_entries = 3
    df = m.to_dataframe(data)
    assert df.shape == (1, initial_entries + added_entries)
    assert df.frequency.iloc[0] == 1.2
    assert df.io_module.iloc[0] is None
    assert df.root_path.iloc[0] is None
    assert df.node_path.iloc[0] is None


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


