"""
This module implements reading and writing of Measurement subclasses to disk using netCDF4.

Each node is a netCDF4 Group;
numpy arrays that are instance attributes are stored as netCDF4 variables;
dicts are stored hierarchically as groups with special names;
other instance attribute are stored as ncattrs of the group.


Limitations and issues.

netCDF4 returns strings as unicode.
On read, all unicode objects are converted to Python2 str type.

netCDF4 returns sequence ncattrs as numpy ndarrays.
On read, all arrays are converted to lists.

netCDF4 cannot store None or boolean types as ncattrs.
These are stored as special strings that are attributes of the IO class, and converted back on read.
This is a little bit gross but probably safe in practice.
"""
import os
import netCDF4
import numpy as np
from kid_readout.measurement import core


class IO(core.IO):

    # These values are used to store None, True, and False as ncattrs.
    none = '__None__'
    true = '__True__'
    false = '__False__'

    # This dictionary translates between numpy complex dtypes and netCDF4 compound types.
    npy_to_netcdf = {np.dtype('complex64'): {'datatype': np.dtype([('real', 'f4'), ('imag', 'f4')]),
                                             'name': 'complex64'},
                     np.dtype('complex128'): {'datatype': np.dtype([('real', 'f8'), ('imag', 'f8')]),
                                              'name': 'complex128'}}

    # Dictionaries are stored as Groups with names that end with this string.
    is_dict = '.dict'

    def __init__(self, root_path):
        self.root_path = os.path.expanduser(root_path)
        try:
            self.root = netCDF4.Dataset(self.root_path, mode='r')
        except RuntimeError:
            self.root = netCDF4.Dataset(root_path, mode='w', clobber=False)

    def close(self):
        try:
            self.root.close()
        except RuntimeError:
            pass

    @property
    def closed(self):
        try:
            return ~self.root.isopen
        except AttributeError:
            raise NotImplementedError("Upgrade netCDF4!")

    def create_node(self, node_path):
        # This will correctly fail to validate an attempt to create the root node with node_path = ''
        core.validate_node_path(node_path)
        existing, new = core.split(node_path)
        self._get_node(existing).createGroup(new)

    def write_array(self, node_path, name, array, dimensions):
        node = self._get_node(node_path)
        if (name,) == dimensions and name not in node.dimensions:
            node.createDimension(name, array.size)
        try:
            npy_datatype = self.npy_to_netcdf[array.dtype]['datatype']
            netcdf_datatype = node.createCompoundType(self.npy_to_netcdf[array.dtype]['datatype'],
                                                      self.npy_to_netcdf[array.dtype]['name'])
        except KeyError:
            npy_datatype = netcdf_datatype = array.dtype
        variable = node.createVariable(name, netcdf_datatype, dimensions)
        variable[:] = array.view(npy_datatype)

    def write_other(self, node_path, key, value):
        node = self._get_node(node_path)
        if isinstance(value, dict):
            self._write_dict_group(node, key, value)
        else:
            setattr(node, key, self._write_convert(value))

    def read_array(self, node_path, name):
        node = self._get_node(node_path)
        nc_variable = node.variables[name]
        return nc_variable[:].view(nc_variable.datatype.name)

    def read_other(self, node_path, name):
        node = self._get_node(node_path)
        if name + self.is_dict in node.groups.keys():
            return self._read_dict_group(node.groups[name + self.is_dict])
        else:
            return self._read_convert(node.__dict__[name])

    def measurement_names(self, node_path):
        node = self._get_node(node_path)
        return [name for name in node.groups.keys() if not name.endswith(self.is_dict)]

    def array_names(self, node_path):
        node = self._get_node(node_path)
        return node.variables.keys()

    def other_names(self, node_path):
        node = self._get_node(node_path)
        return node.ncattrs() + [name.replace(self.is_dict, '') for name in node.groups.keys()
                                 if name.endswith(self.is_dict)]

    # Private methods.

    def _get_node(self, node_path):
        node = self.root
        if node_path != '':
            core.validate_node_path(node_path)
            for name in core.explode(node_path):
                node = node.groups[name]
        return node

    def _write_dict_group(self, node, name, dictionary):
        dict_group = node.createGroup(name + self.is_dict)
        for k, v in dictionary.items():
            if isinstance(v, dict):
                self._write_dict_group(dict_group, k, v)
            else:
                if isinstance(v, bool):
                    v = int(v)
                setattr(dict_group, k, v)

    def _read_dict_group(self, group):
        items = [(k, self._read_convert(v)) for k, v in group.__dict__.items()]
        return dict(items + [(name.replace(self.is_dict, ''), self._read_dict_group(group))
                             for name, group in group.groups.items()])

    def _write_convert(self, obj):
        if obj is True:
            return self.true
        elif obj is False:
            return self.false
        elif obj is None:
            return self.none
        else:
            return obj

    def _read_convert(self, obj):
        if obj == self.none:
            return None
        if isinstance(obj, unicode):
            return str(obj)
        elif isinstance(obj, np.ndarray):
            return list(obj)
        else:
            return obj
