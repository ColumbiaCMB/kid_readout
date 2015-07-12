from __future__ import division
import os
import time
from collections import OrderedDict, namedtuple
import netCDF4
import numpy as np
import pandas as pd


class DataFile(object):

    complex64 = np.dtype([('real', 'f4'), ('imag', 'f4')])
    complex128 = np.dtype([('real', 'f8'), ('imag', 'f8')])

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


class Measurement(object):
    """
    This is an abstract class that represents a measurement for a single channel.

    Measurements are hierarchical: a Measurement can contain other Measurements.

    Each Measurement should be self-contained, meaning that it should contain all data and metadata necessary to
    analyze and understand it. Can this include temperature data?

    Caching: all raw data attributes are public and all special or processed data attributes are private.
    """

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

    def __init__(self, state=None, analyze=False):
        self._parent = None
        if state is None:
            self.state = {}
        if analyze:
            self.analyze()

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

    def analyze(self):
        """
        analyze the raw data and create all data products.
        :return: None
        """
        pass

    def to_dataframe(self):
        """
        :return: a DataFrame containing all of the instance attributes.
        """
        pass
