"""
This module implements reading and writing of Measurement subclasses to disk using numpy ndarrays.
"""
import os
import netCDF4
import numpy as np
from kid_readout.measurement import core


npy_to_netcdf = {np.dtype('complex64'): {'datatype': np.dtype([('real', 'f4'), ('imag', 'f4')]),
                                         'name': 'complex64'},
                 np.dtype('complex128'): {'datatype': np.dtype([('real', 'f8'), ('imag', 'f8')]),
                                          'name': 'complex128'}}


def create(filename):
    return netCDF4.Dataset(os.path.expanduser(filename), mode='w', clobber=False)


def new(group, name):
    return group.createGroup(name)


def write(thing, group, name):
    if isinstance(thing, np.ndarray):
        dimension = group.createDimension(name, thing.size)
        try:
            npy_datatype = npy_to_netcdf[thing.dtype]['datatype']
            netcdf_datatype = group.createCompoundType(npy_to_netcdf[thing.dtype]['datatype'],
                                                       npy_to_netcdf[thing.dtype]['name'])
        except KeyError:
            npy_datatype = netcdf_datatype = thing.dtype
        variable = group.createVariable(name, netcdf_datatype, (name,))
        variable[:] = thing.view(npy_datatype)
    elif isinstance(thing, dict):
        dict_group = group.createGroup(name)
        setattr(dict_group, core.CLASS_NAME, 'dict')
        for k, v in thing.items():
            write(v, dict_group, k)
    else:
        setattr(group, name, thing)


def read(top):
    return _visit(None, top)


def _visit(parent, location):
    class_name = getattr(location, core.CLASS_NAME)
    if class_name == 'dict':
        current = _visit_dict(location)
    else:
        class_ = core.get_class(class_name)
        names = location.groups.keys()  # NB: keep unicode
        if core.is_sequence(class_):
            current = class_([_visit(None, location.groups[name])
                              for name in sorted(names, key=int)])
            for meas in current:
                meas._parent = parent  # Preserve the current convention.
        else:
            current = class_()
            for name in names:
                setattr(current, name, _visit(current, location.groups[name]))
        current._parent = parent
        # Load variables
        variable_names = location.variables.keys()
        for variable_name in variable_names:
            variable = location.variables[variable_name]
            setattr(current, variable_name, variable[:].view(variable.datatype.name))
        # Load everything else
        for k, v in location.__dict__.items():
            setattr(current, k, v)
    return current


def _visit_dict(group):
    # Note that
    # k is measurement.CLASS_NAME == False
    # because netCDF4 returns all strings as unicode.
    return dict([(k, v) for k, v in group.__dict__.items() if k != core.CLASS_NAME] +
                [(name, _visit_dict(group)) for name, group in group.groups.items()])



"""
class Writer(measurement.Writer):



    def __init__(self, filename):

    # These dictionaries contain all the metadata necessary for reading and writing netCDF4 files.
    # Each subclass of Measurement must describe the attributes that will be written to disk.

    # The keys are names of netCDF4 dimensions and the values are integers or None, for an unlimited dimension.
    netcdf_dimensions = OrderedDict()

    # The keys are the names of the class attributes that are written to disk, and the values are Variables objects that
    # contain the netCDF4 metadata.
    netcdf_variables = OrderedDict()

    # netcdf_name is the name of the variable in the netCDF4 file; it should be the same as the class attribute
    # data_type is the data type of the variable; this is necessary to handle complex data.
    # dimensions is a tuple of strings that are the names of dimensions defined in netcdf_dimensions; attributes that
    # are scalars use the empty tuple ().
    Variable = namedtuple('Variable', ('netcdf_name', 'data_type', 'dimensions'))



    # From old Measurement class
    def to_group(self, group, numpy_to_netcdf):
        for dimension, size in self.netcdf_dimensions.items():
            group.createDimension(dimension, size)
        for attribute, (netcdf_name, data_type, dimensions) in self.netcdf_variables.items():
            netcdf_data_type = numpy_to_netcdf.get(data_type, data_type)
            group.createVariable(netcdf_name, netcdf_data_type, dimensions=dimensions)
            if not dimensions:  # A scalar variable has no dimensions, and the empty tuple evaluates to False.
                group.variables[netcdf_name].assignValue(getattr(self, attribute))
            else:  # Cast an array to the netcdf data type and view it as the appropriate compound type, if necessary.
                group.variables[netcdf_name][:] = getattr(self, attribute).astype(data_type).view(netcdf_data_type)
        state = group.createGroup('state')
        for key, value in self.state.items():
            setattr(state, key, value)

    def from_group(self, group):
        for attribute, (netcdf_name, data_type, dimension_tuple) in self.netcdf_variables.items():
            nc_variable = group.variables[netcdf_name]
            if not dimension_tuple:  # A scalar variable has no dimensions, and the empty tuple evaluates to False.
                # TODO: keep an eye on this: getValue() is supposed to return the scalar value but returns an array.
                setattr(self, attribute, nc_variable.getValue()[0])
            else:  # Convert an array from a compound type to standard type, if necessary.
                setattr(self, attribute, nc_variable[:].view(nc_variable.datatype.name))
        self.state = dict(group.groups['state'].__dict__)
        return self

    # From measurement.Stream
    netcdf_dimensions = OrderedDict(s21=None)

    netcdf_variables = OrderedDict(frequency=Measurement.Variable(netcdf_name='frequency',
                                                                  data_type=np.float64,
                                                                  dimensions=()),
                                   s21=Measurement.Variable(netcdf_name='s21',
                                                            data_type=np.complex64,
                                                            dimensions=('s21',)),
                                   start_epoch=Measurement.Variable(netcdf_name='start_epoch',
                                                                    data_type=np.float64,
                                                                    dimensions=()),
                                   end_epoch=Measurement.Variable(netcdf_name='end_epoch',
                                                                  data_type=np.float64,
                                                                  dimensions=()))

    # From measurement.Sweep
    netcdf_dimensions = OrderedDict(frequency=None)
    netcdf_variables = {'frequency': Measurement.Variable(netcdf_name='frequency',
                                                          data_type=np.float64,
                                                          dimensions=('frequency',)),
                        's21': Measurement.Variable(netcdf_name='s21',
                                                    data_type=np.complex64,
                                                    dimensions=('frequency',))}

    # from measurement.FrequencySweep
    def to_group(self, group, numpy_to_netcdf):
        super(FrequencySweep, self).to_group(group, numpy_to_netcdf)
        streams_group = group.createGroup('streams')
        for n, stream in enumerate(self.streams):
            stream_group = streams_group.createGroup("Stream_{}".format(n))
            stream.to_group(stream_group, numpy_to_netcdf)

    def from_group(self, group):
        super(FrequencySweep, self).from_group(group)
        stream_list = []
        for stream_group in group.groups['streams'].groups.values():
            stream = Stream().from_group(stream_group)
            stream._parent = self
            stream_list.append(stream)
        self.streams = MeasurementTuple(stream_list)
        return self

    # From measurement.SweepStream
    def to_group(self, group, numpy_to_netcdf):
        super(SweepStream, self).to_group(group, numpy_to_netcdf)
        sweep_group = group.createGroup('sweep')
        self.sweep.to_group(sweep_group, numpy_to_netcdf)
        stream_group = group.createGroup('stream')
        self.stream.to_group(stream_group, numpy_to_netcdf)

    def from_group(self, group):
        super(SweepStream, self).from_group(group)
        self.sweep = ResonatorSweep().from_group(group.groups['sweep'])
        self.sweep._parent = self
        self.stream = Stream().from_group(group.groups['stream'])
        self.sweep._parent = self
        return self


class DataFile(object):

    #            self.filename = '{}{}.nc'.format(time.strftime('%Y-%m-%d_%H%M%S'), suffix)
    def __init__(self, filename):
        self.filename = os.path.expanduser(filename)
        if os.path.isfile(self.filename):
            mode = 'r'
        else:
            mode = 'w'
        self.dataset = netCDF4.Dataset(self.filename, mode=mode)

    @property
    def measurements(self):
        return self.dataset.groups

    def save_measurement(self, measurement, name=None):
        if name is None:
            name = '{}_{}'.format(measurement.__class__.__name__, len(self.dataset.groups))
        group = self.dataset.createGroup(name)
        group.class_ = measurement.__class__.__name__
        cdf64 = group.createCompoundType(self.complex64, 'complex64')
        cdf128 = group.createCompoundType(self.complex128, 'complex128')
        measurement.to_group(group, numpy_to_netcdf={np.complex64: cdf64, np.complex128: cdf128})
        self.dataset.sync()

    # Implement automatic loading by class
    def load_measurement(self, measurement_name):
        measurement_group = self.measurements[measurement_name]
        return globals()[measurement_group.class_]().from_group(measurement_group)


"""