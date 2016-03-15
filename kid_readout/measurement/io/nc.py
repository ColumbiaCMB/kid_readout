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


# TODO: use full dict/list names everywhere internally
class IO(core.IO):

    # These special strings are used to store None, True, and False as ncattrs.
    on_write = {None: '_None',
                True: '_True',
                False: '_False'}

    on_read = {'_None': None,
               '_True': True,
               '_False': False}

    # This dictionary translates between numpy complex dtypes and netCDF4 compound types.
    npy_to_netcdf = {np.dtype('complex64'): {'datatype': np.dtype([('real', 'f4'), ('imag', 'f4')]),
                                             'name': 'complex64'},
                     np.dtype('complex128'): {'datatype': np.dtype([('real', 'f8'), ('imag', 'f8')]),
                                              'name': 'complex128'}}

    # Dictionaries are stored as Groups with names that end with this string.
    is_dict = '.dict'
    # Sequences that are not explicitly declared as arrays with their own dimensions are stored as Variables with names
    # that end with this string, and are returned on read as lists.
    is_list = '.list'

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
        """
        Write the given array to the node at node_path with the given name and dimensions.

        When writing arrays to a node, each dimension is created the first time it appears in the dimensions tuple.
        The dimension is created with size equal to the corresponding dimension of the given array. Thus, there is no
        restriction on the order in which arrays are written. Writing will still fail if two arrays share a dimension
        name and have different shape along the corresponding axes. Since this would have caused
        Measurement._validate_dimensions() to fail, this should not happen unless array sizes are modified after
        instantiation somehow.

        :param node_path: the node path as a string.
        :param name: the name of the variable.
        :param array: the array containing the data.
        :param dimensions: a tuple of strings with the dimensions that correspond to the dimensions of the array.
        :return: None.
        """
        node = self._get_node(node_path)
        for n, dimension in enumerate(dimensions):
            if dimension not in node.dimensions:
                node.createDimension(dimension, array.shape[n])
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
        self._write_to_group(node, key, value)

    def read_array(self, node_path, name):
        node = self._get_node(node_path)
        nc_variable = node.variables[name]
        return nc_variable[:].view(nc_variable.datatype.name)

    def read_other(self, node_path, name):
        node = self._get_node(node_path)
        if name + self.is_dict in node.groups:
            return self._read_dict(node.groups[name + self.is_dict])
        elif name + self.is_list in node.variables:
            return self._read_sequence(node, name + self.is_list)
        else:
            value = node.__dict__[name]
            return self.on_read.get(value, value)

    def measurement_names(self, node_path):
        node = self._get_node(node_path)
        return [name for name in node.groups if not name.endswith(self.is_dict)]

    def array_names(self, node_path):
        node = self._get_node(node_path)
        return [key for key in node.variables if not key.endswith(self.is_list)]

    def other_names(self, node_path):
        node = self._get_node(node_path)
        dicts = [name.replace(self.is_dict, '') for name in node.groups if name.endswith(self.is_dict)]
        lists = [name.replace(self.is_list, '') for name in node.variables if name.endswith(self.is_list)]
        return node.ncattrs() + lists + dicts

    # Private methods.

    def _get_node(self, node_path):
        node = self.root
        if node_path != '':
            core.validate_node_path(node_path)
            for name in core.explode(node_path):
                node = node.groups[name]
        return node

    def _write_to_group(self, group, key, value):
        """
        This method directly writes non-container values to the given Group or calls the appropriate function to
        write container values.

        :param group: the netCDF4 Group.
        :param key: the external name of the value to write, with no special suffix.
        :param value: the value to write
        :return: None.
        """
        if isinstance(value, dict):
            self._write_dict(group, key + self.is_dict, value)
        elif isinstance(value, (list, tuple, np.ndarray)):
            self._write_sequence(group, key + self.is_list, value)
        else:
            setattr(group, key, self.on_write.get(value, value))

    def _write_sequence(self, group, key, value):
        """
        Write the given sequence (value) to the given netCDF4 Group using the given name (key).

        It attempts to determine the appropriate data type to use by creating a numpy array from the sequence and
        examining its dtype. This should succeed for sequences that contain only numbers, only strings or unicode, or
        only booleans. Obviously, this will result in some type conversion for mixed arrays, and will fail completely
        in some cases. Note that netCDF4 will not accept None as a value in an array.

        :param group: the netCDF Group.
        :param key: the name of the dimension and variable to use for storing the sequence, ending with self.is_list.
        :param value: the sequence to store.
        :return: None.
        """
        array = np.array(value)
        if array.dtype.type in (np.unicode_, np.str_):
            group.createDimension(key, array.size)
            variable = group.createVariable(key, str, key)  # This creates a variable-length string array.
            variable[:] = array.astype(np.object)
        elif array.dtype.type is np.bool_:  # This seems to be True only if all elements are bool
            group.createDimension(key, array.size)
            variable = group.createVariable(key, str, key)  # See above; booleans are stored as strings.
            variable[:] = np.array([self.on_write[obj] for obj in array], dtype=np.object)
        else:
            group.createDimension(key, array.size)
            variable = group.createVariable(key, array.dtype, key)
            variable[:] = array

    def _read_sequence(self, group, key):
        """
        Return a list containing the stored sequence with the given name (key) from the given netCDF4 Group.

        :param group: the netCDF4 Group to read from.
        :param key: the name of the Variable to read, ending with self.is_list.
        :return: a list containing the contents of the Variable.
        """
        try:
            array = group.variables[key][:]
        except IndexError:  # An empty Variable raises an IndexError
            return []
        try:
            return [self.on_read[v] for v in array]
        except KeyError:
            return list(array)

    def _write_dict(self, group, dict_name, dictionary):
        """
        Create a new Group with the given name and write the given dictionary to it.

        :param group: the netCDF4 Group.
        :param dict_name: the name of the dictionary, ending with self.is_dict.
        :param dictionary: the dict to write.
        :return: None.
        """
        dict_group = group.createGroup(dict_name)
        for k, v in dictionary.items():
            self._write_to_group(dict_group, k, v)

    def _read_dict(self, group):
        ncattrs = [(k, self.on_read.get(v, v)) for k, v in group.__dict__.items()]
        list_names = [name for name in group.variables if name.endswith(self.is_list)]
        lists = [(list_name.replace(self.is_list, ''), self._read_sequence(group, list_name))
                 for list_name in list_names]
        dict_names = [name for name in group.groups if name.endswith(self.is_dict)]
        dicts = [(dict_name.replace(self.is_dict, ''), self._read_dict(group.groups[dict_name]))
                 for dict_name in dict_names]
        return dict(ncattrs + lists + dicts)
