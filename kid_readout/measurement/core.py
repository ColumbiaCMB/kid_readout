"""
This module is the core of the measurement subpackage.

The main ideas are (1) to provide data container classes that define a format and include basic analysis code for the
data, and (2) to separate these data container classes from the format in which the data are stored on disk. These
goals are implemented using a Measurement class and a few functions and auxiliary classes.

The format is designed to handle hierarchical measurements naturally. For example, a frequency sweep is a collection
of streams of time-ordered data, so the both the class structure in memory and the hierarchical structure on disk are
a standard tree. This structure allows measurements that follow the specified format to be saved to disk and
re-created later without any additional metadata.

To create a new measurement, simply write a subclass of Measurement. Public attributes of the class will
automatically be saved to disk by the write() function and re-instantiated by the read() function. If a class
contains attributes that are either measurements or sequences of measurements (that use the provided MeasurementList
and MeasurementTuple containers), these will also be saved and restored correctly. Each Measurement should be
self-contained, meaning that it should contain all data and metadata necessary to analyze and understand it.

A measurement will typically contain arrays that contain most of the data and a state dictionary that holds metadata.
Because of restrictions imposed by the libraries used to store data on disk, the dictionary cannot hold arbitrary
values. See the Measurement docstring for details.

Because the same measurement class can potentially describe many measurements that use different equipment and thus
have different state data, the best way to handle presence or absence of equipment or settings is simply through the
presence or absence of entries in the state dictionary. It's easy for external code to check
if 'some_equipment_info' in state: analyze_it()
to determine whether to analyze the information. If an entry has to be present for some reason, using None is better
than NaN to signify "not available," even if the state value is usually a number. An example of a good reason would
be if some equipment was used during the measurement, but its state was not available when the measurement was
written. Besides the advantage of avoiding comparison subtleties, None will cause numeric operations to blow up
immediately.

Except for the root node, every node in the tree contains a measurement or a sequence of measurements. (The root node
is reserved for metadata.) Every node can thus be reached by a unique node path. For example, if the root contained
only one Sweepstream measurement with the name "sweepstream," the structure would be

sweepstream
  stream
  sweep
    streams
      0
      1
      etc.

so the node path to the main SweepStream would be just "sweepstream" while the node path to one of the streams in the
sweep could be "sweepstream:sweep:stream:1" and so on. The node paths can be used to address the data structure in a
format-independent way in order to save new data in specific locations or to load subsets of the measurement tree.
"""
import re
import inspect
import warnings
import importlib
import numpy as np

CLASS_NAME = '_class'  # This is the string used by writer objects to save class names.

# These names cannot be used for attributes because they are used as part of the public DataFrame interface.
IO_CLASS_NAME = 'io_class'  # This is the fully-qualified name of the io class used to read a measurement from disk.
IO_MODULE = 'io_module'  # This is the full-qualified name of the module used to read and write legacy data.
ROOT_PATH = 'root_path'  # This is the root file or directory from which a measurement was read from disk.
NODE_PATH = 'node_path'  # This is the node path from the root node to the measurement node.

# TODO: decide which names really need to be reserved.
RESERVED_NAMES = [CLASS_NAME, IO_CLASS_NAME, IO_MODULE, ROOT_PATH, NODE_PATH]
NODE_PATH_SEPARATOR = ':'


class Measurement(object):
    """
    This is an abstract class that represents a measurement.

    To create a new measurement, simply write a subclass of Measurement that obeys the restrictions described here.

    Hierarchy.
    Measurements that are public attributes of a measurement will be automatically saved and loaded.
    Tuples or lists of measurements must be contained in MeasurementTuple or MeasurementList objects, which must also be
    public attributes.

    Instantiation.
    When a measurement is read from disk, the read() function inspects the signature of its __init__() method and
    calls it with the saved values of the corresponding variables. This allows every measurement read from disk to
    re-initialize itself and validate its data. However, because only public attributes are saved, this creates the
    restriction that a measurement must save the values with which it is initialized and cannot discard them or store
    them internally under other names. For example, if a measurement's __init__() method contained an argument foo
    that is stored as self.bar, the value in bar will be saved to disk as bar but the read() function will not be
    able to tell that this value should be passed as the value of parameter foo. See the instantiate() function.

    Arrays.
    Each measurement has a dimensions class attribute that contains metadata for its array dimensions. This is
    necessary for the netCDF4 IO class to handle the array dimensions correctly, and it also allows the classes to
    check the dimensions of their arrays on instantiation through the _validate_dimensions() method. The format of the
    an entry in the dimensions dict is 'array_name': dimension_tuple, where dimension tuple is a tuple of strings
    that are the names of the dimensions. To pass validation, each dimension name must correspond to an attribute or
    property of the class that is a 1-D array with size equal to the corresponding element of array.shape. Thus, arrays
    that have a given dimension must all have the same length along that dimension. For example, if there is an entry
    's21_raw': ('time', 'frequency)
    and another entry
    'frequency': ('frequency',)
    then
    s21_raw.shape[1] == frequency.size
    must be True. If the array corresponding to some dimension is not intended to be saved, it can be implemented as a
    property. For example, in the case above, the 'time' dimension could be implemented as a property. The instance
    would still pass validation as long as
    s21_raw.shape[0] == time.size
    were True.

    Containers.
    Measurements store state information in a dictionary. (They actually use a subclass called StateDict, which has
    extra access features and validation.) Implementations of the IO class must be able to store the following objects:
    -basic types, such as numbers, booleans, strings, booleans, and None;
    -sequences, i.e. lists, tuples, and arrays, that contain exactly one of the above basic types except None; sequences
     cannot contain other sequences, and all sequences are returned as lists on read.
    -dictionaries whose keys are strings and whose values are dictionaries, sequences, or basic types; the contained
     dictionaries and sequences have the same requirements as above.
    Classes that do not obey these restrictions may fail to save or may have missing or corrupted data when read from
    disk. Most of these restrictions come from the limitations of netCDF4 and json. Some of them could be relaxed
    with more work, if necessary.

    Two values that are particularly problematic are None and NaN. None requires a special case because netCDF4
    cannot write it, so it is allowed as an attribute value or dictionary value but not in sequences. NaN is stored
    and retrieved correctly but is not recommended as an indicator of missing state because it causes comparisons to
    fail: float('NaN') == float('NaN') is always False. See Measurement.__eq__() for more.
    """

    dimensions = {}

    def __init__(self, state=None, analyze=False, description='Measurement'):
        """
        Return a new Measurement instance.

        :param state: a dictionary of state information.
        :param analyze: if True (default False), the class will call its analyze() method after instantiation.
        :param description: a string describing the measurement.
        :return: a new Measurement instance.
        """
        if state is not None:
            state = to_state_dict(state)
        self.state = state
        self.description = description
        self._parent = None
        self._io_class = None
        self._root_path = None
        self._node_path = None
        self._validate_dimensions()
        if analyze:
            self.analyze()

    @property
    def class_name(self):
        return self.__class__.__name__

    def nodes(self):
        if self._parent is None:
            return []
        else:
            return self._parent.nodes() + [self._parent.locate(self)]

    def analyze(self):
        """
        Analyze the raw data and create all data products.

        :return: None
        """
        pass

    def to_dataframe(self):
        """
        Return a pandas DataFrame containing data from this Measurement.

        This method should return state information and analysis products, such as fit parameters, but not large objects
        like time-ordered data.

        :return: a DataFrame containing data from this Measurement.
        """
        pass

    def add_origin(self, dataframe):
        """
        Add to the given dataframe enough information to load the data from which it was created. Using this
        information, the from_series() function in this module will return the original data.

        This method adds the IO class, the path to the root file or directory, and the node path corresponding to this
        measurement, which will all be None unless the Measurement was created from files on disk.
        """
        dataframe['io_class'] = self._io_class
        dataframe['root_path'] = self._root_path
        dataframe['node_path'] = self._node_path

    # TODO: add timestream or sweep indices?
    def add_legacy_origin(self, dataframe):
        """
        Add to the given dataframe information about the origin of the data. It's going to be difficult to implement
        automatic loading of original data, but at least this will record the netCDF4 file.
        """
        dataframe['io_module'] = 'kid_readout.measurement.legacy'
        dataframe['root_path'] = self._root_path

    def _validate_dimensions(self):
        for name, dimension_tuple in self.dimensions.items():
            if not getattr(self, name).shape == tuple(getattr(self, dimension).size for dimension in dimension_tuple):
                raise ValueError("Shape of {} does not match size of {}.".format(name, dimension_tuple))

    def _locate(self, obj):
        for key, value in self.__dict__.items():
            if obj is value:
                return str(key)

    def __eq__(self, other):
        """
        Recursively compare two measurements. At each level, the function tests that both instances have the same public
        attributes (meaning those that do not start with an underscore), that all these attributes are equal,
        and that the classes of the measurements are equal. Because the data we store mixes booleans and numbers,
        boolean values stored as attributes are compared using identity, not equality. Note that this is not done
        within containers.

        The function does not compare private attributes, and does not even check whether the instances have the same
        private attributes.

        :param other: a Measurement instance.
        :return: True if self compares equal with other, and False if not.
        """
        try:
            keys_s = ['__class__'] + [k for k in self.__dict__ if not k.startswith('_')]
            keys_o = ['__class__'] + [k for k in other.__dict__ if not k.startswith('_')]
            assert set(keys_s) == set(keys_o)
            for key in keys_s:
                value_s = getattr(self, key)
                value_o = getattr(other, key)
                if issubclass(value_s.__class__, Measurement):
                    assert value_s.__eq__(value_o)
                elif issubclass(value_s.__class__, MeasurementList):
                    assert len(value_s) == len(value_o)
                    for meas_s, meas_o in zip(value_s, value_o):
                        assert meas_s.__eq__(meas_o)
                elif isinstance(value_s, np.ndarray):  # This allows declared arrays to contain NaN and be equal.
                    assert np.all(np.isnan(value_s) == np.isnan(value_o))
                    assert np.all(value_s[~np.isnan(value_s)] == value_o[~np.isnan(value_o)])
                else:  # This will fail for NaN or sequences that contain any NaN values.
                    assert value_s == value_o
                    if isinstance(value_s, bool) or isinstance(value_o, bool):
                        assert value_s is value_o
        except AssertionError:
            return False
        return True


class MeasurementList(list):
    """
    Measurements containing lists of Measurements must use instances of this class so that loading and saving are
    handled correctly.
    """

    def __init__(self, *args):
        self._parent = None
        super(MeasurementList, self).__init__(*args)

    @property
    def class_name(self):
        return self.__class__.__name__

    def _locate(self, obj):
        for index, value in enumerate(self):
            if obj is value:
                return str(index)

    def nodes(self):
        if self._parent is None:
            return []
        else:
            return self._parent.nodes() + [self._parent._locate(self)]

    def append(self, item):
        item._parent = self
        super(MeasurementList, self).append(item)

    def extend(self, iterable):
        for i in iterable:
            self.append(i)

    def insert(self, index, p_object):
        p_object._parent = self
        super(MeasurementList, self).insert(index, p_object)

    def __setitem__(self, key, value):
        value._parent = self
        super(MeasurementList, self).__setitem__(key, value)


class IOList(MeasurementList):

    def __init__(self, io, root_name):
        self.io = io
        self.root_name = root_name
        self._len = 0
        super(MeasurementList, self).__init__()

    @property
    def class_name(self):
        return self.__class__.__base__.__name__

    def append(self, item):
        node_list = [self.root_name] + self.nodes() + [str(len(self))]
        self.io.write(item, join(*node_list))
        self._len += 1

    def extend(self, iterable):
        for i in iterable:
            self.append(i)

    def insert(self, index, p_object):
        raise NotImplementedError()

    def remove(self, value):
        raise NotImplementedError()

    def pop(self, index=None):
        raise NotImplementedError()

    def index(self, value, start=None, stop=None):
        raise NotImplementedError()

    def count(self, value):
        raise NotImplementedError()

    def sort(self, cmp=None, key=None, reverse=False):
        raise NotImplementedError()

    def reverse(self):
        raise NotImplementedError()

    def __iter__(self):
        """
        Instances of this class appear like empty lists, so iteration stops immediately.

        :return: an empty iterator.
        """
        return iter(())

    def __len__(self):
        return self._len

    def __getitem__(self, item):
        raise NotImplementedError()

    def __setitem__(self, key, value):
        raise NotImplementedError()


class MeasurementError(Exception):
    """
    Raised for module-specific errors.
    """
    pass


class StateDict(dict):
    """
    This class adds attribute access and some content restrictions to the dict class.

    Measurements that require dictionaries can use the function to_state_dict() to perform partial validation of a
    dictionary to ensure that the data can be re-created properly from disk.
    """

    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__
    __delattr__ = dict.__delitem__
    __copy__ = lambda self: StateDict(self)
    __getstate__ = lambda: None
    __slots__ = ()


# TODO: incorporate restrictions into __init__().
# TODO: implement more error checking.
def to_state_dict(dictionary):
    if not all([isinstance(k, (str, unicode)) for k in dictionary]):
        raise MeasurementError("Dictionary keys must be strings.")
    dicts = [(k, to_state_dict(v)) for k, v in dictionary.items() if isinstance(v, dict)]
    others = [(k, v) for k, v in dictionary.items() if not isinstance(v, dict)]
    return StateDict(dicts + others)


class IO(object):
    """
    This is an abstract class that specifies the IO interface.

    Implementations should be able to store large numpy arrays efficiently.
    """

    def __init__(self, root_path):
        """
        Return a new IO object that will read to or write from the given root directory or file. If the root does not
        exist, it should be created. Implementations should NEVER open an existing file in write mode, and in general
        should make it impossible to overwrite data. Appending to existing files may be useful.

        :param root_path: the path to the root directory or file.
        :return: a new IO object that can read from and write to the root object at the given path.
        """
        self.root_path = root_path
        self.root = None

    def close(self):
        """
        Close open files.
        """
        pass

    @property
    def closed(self):
        """
        Return True if all files on disk are closed.
        """
        return True

    def default_name(self, measurement):
        """
        Return a name for the given measurement class or instance that is guaranteed to be unique at the root level.

        :param measurement: a Measurement subclass or instance.
        :return: a string consisting of the class name and a number that is one plus the number of measurements
          already stored at root level.
        """
        return measurement.class_name + str(len(self.measurement_names()))

    def write(self, measurement, node_path=None):
        """
        Write the measurement to disk at the given node path.

        :param measurement: the measurement instance to write to disk.
        :param node_path: the node_path to the node that will contain this object; all but the final node in node_path must
        already exist.
        :return: None
        """
        if node_path is None:
            node_path = self.default_name(measurement)
        self._write_node(measurement, node_path)

    def read(self, node_path, translate=None):
        """
        Read a measurement from disk and return it.

        :param node_path:the path to the node to be loaded, in the form 'node0:node1:node2'
        :param translate: a dictionary with entries 'original_class': 'new_class'; class names must be fully-qualified.
        :return: the measurement corresponding to the given node.
        """
        if translate is None:
            translate = {}
        return self._read_node(node_path, translate)

    # These functions are used by

    def create_node(self, node_path):
        """
        Create a node at the end of the given path; all but the final node in the path must already exist.
        """
        pass

    def write_other(self, node_path, key, value):
        """
        Write value to node_path with name key; value should not be a numpy array.
        """
        pass

    def write_array(self, node_path, key, value, dimensions):
        """
        Write value, a numpy array, to node_path with name key.
        """
        pass

    def read_array(self, node_path, key):
        """
        Read array key from node_path.
        """
        pass

    def read_other(self, node_path, key):
        """
        Read non-array object with name key from node_path.
        """
        pass

    def measurement_names(self, node_path=''):
        """
        Return the names of all measurements contained in the measurement at node_path.
        """
        pass

    def array_names(self, node_path):
        """
        Return the names of all arrays contained in the measurement at node_path.
        """
        pass

    def other_names(self, node_path):
        """
        Return the names of all other variables contained in the measurement at node_path.
        """
        pass

    def __getattr__(self, item):
        if item in self.measurement_names():
            return self.read(item)
        else:
            raise AttributeError()

    """
    def __getitem__(self, item):
        try:
            return self.read(self.measurement_names()[item])
        except IndexError:
            raise KeyError()
    """

    # TODO: add methods?
    def __dir__(self):
        return self.__dict__.keys() + self.measurement_names()

    def _write_node(self, measurement, node_path):
        """
        Create a new node at the given node path using the given IO instance and write the measurement data to it.

        :param measurement: a Measurement instance.
        :param node_path: the path of the new node into which the measurement will be written.
        :return: None
        """
        self.create_node(node_path)
        this_class_name = '{}.{}'.format(measurement.__module__, measurement.class_name)
        self.write_other(node_path, CLASS_NAME, this_class_name)
        items = [(key, value) for key, value in measurement.__dict__.items()
                 if not key.startswith('_') and key not in RESERVED_NAMES and key not in measurement.dimensions]
        for key, value in items:
            if isinstance(value, Measurement):
                self._write_node(value, join(node_path, key))
            elif isinstance(value, MeasurementList):
                sequence_node_path = join(node_path, key)
                self.create_node(sequence_node_path)
                sequence_class_name = '{}.{}'.format(value.__module__, value.class_name)
                self.write_other(sequence_node_path, CLASS_NAME, sequence_class_name)
                for index, meas in enumerate(value):
                    self._write_node(meas, join(sequence_node_path, str(index)))
            else:
                self.write_other(node_path, key, value)
        # Saving arrays in order allows the netCDF group to create the dimensions.
        for array_name, dimensions in measurement.dimensions.items():
            self.write_array(node_path, array_name, getattr(measurement, array_name), dimensions)

    def _read_node(self, node_path, translate):
        original_class_name = self.read_other(node_path, CLASS_NAME)
        class_name = translate.get(original_class_name, original_class_name)
        measurement_names = self.measurement_names(node_path)
        if is_sequence(class_name):
            # Use the name of each measurement, which is an int, to restore the order in the sequence.
            contents = [self._read_node(join(node_path, measurement_name), translate)
                        for measurement_name in sorted(measurement_names, key=int)]
            current = instantiate_sequence(class_name, contents)
        else:
            variables = {}
            for measurement_name in measurement_names:
                variables[measurement_name] = self._read_node(join(node_path, measurement_name), translate)
            array_names = self.array_names(node_path)
            for array_name in array_names:
                variables[array_name] = self.read_array(node_path, array_name)
            other_names = [vn for vn in self.other_names(node_path)]
            for other_name in other_names:
                variables[other_name] = self.read_other(node_path, other_name)
            current = instantiate(class_name, variables)
        current._io_class = '{}.{}'.format(self.__module__, self.__class__.__name__)
        current._root_path = self.root_path
        current._node_path = node_path
        return current


# Public functions

def write(measurement, io, node_path):
    """
    Write a measurement to disk using the given IO class. This function feeds data to this class and tells it when to
    create new nodes to match the hierarchy of measurements.

    :param measurement: the measurement instance to write to disk.
    :param io: an instance of a class that implements the io interface.
    :param node_path: the node_path to the node that will contain this object; all but the final node in node_path must
      already exist.
    :return: None
    """
    warnings.warn(DeprecationWarning("Use IO.write() instead."))
    io._write_node(measurement, node_path)


def read(io, node_path, translate=None):
    """
    Read a measurement from disk and return it.

    :param io: an instance of a class that implements the IO interface.
    :param node_path:the path to the node to be loaded, in the form 'node0:node1:node2'
    :param translate: a dictionary with entries 'original_class': 'new_class'; all class names must be fully-qualified.
    :return: the measurement corresponding to the given node.
    """
    warnings.warn(DeprecationWarning("Use IO.read() instead."))
    if translate is None:
        translate = {}
    measurement = io._read_node(node_path, translate)
    return measurement


# Class-related functions


def get_class(full_class_name):
    module_name, class_name = full_class_name.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def instantiate(full_class_name, variables):
    """
    Import and instantiate a class using the data in the given dictionary.

    :param full_class_name: the fully-qualified class name as a string, e.g. 'kid_readout.measurement.core.Measurement'.
    :param variables: a dictionary whose keys are the names of the variables available for instantiation and whose
      values are the corresponding values; it must include entries for all non-keyword arguments to __init__().
    :return: an instance of full_class_name instantiated using the given variables.
    """
    class_ = get_class(full_class_name)
    args, varargs, keywords, defaults = inspect.getargspec(class_.__init__)
    arg_values = []
    for arg, default in zip(reversed(args), reversed(defaults)):
        arg_values.append(variables.get(arg, default))
    for arg in reversed(args[1:-len(defaults)]):  # The first arg is 'self'
        arg_values.append(variables[arg])
    instance = class_(*reversed(arg_values))
    return instance


def instantiate_sequence(full_class_name, contents):
    return get_class(full_class_name)(contents)


def is_sequence(full_class_name):
    return issubclass(get_class(full_class_name), MeasurementList)


def from_series(series):
    io = get_class(series.io_class)(series.root_path)
    return io.read(series.node_path)


# Node-related functions


def join(*nodes):
    """
    Join the given nodes using the node path separator, like os.path.join(). Extra separators will be stripped.

    :param nodes: the nodes to join.
    :return: a string representing the joined path.
    """
    return NODE_PATH_SEPARATOR.join([node.strip(NODE_PATH_SEPARATOR) for node in nodes])


def split(node_path):
    """
    Split the given path into a (head, tail) tuple, where tail is the last node in the path and head is the rest, like
    os.path.split(). For example,
    split('one:two:three')
    returns
    ('one:two', 'three')
    If the path contains a single node, then head is the empty string and tail is that node.

    :param node_path: the node path to split.
    :return: a (head, tail) tuple.
    """
    if not node_path:
        return node_path
    else:
        exploded = explode(node_path)
        return join(*exploded[:-1]), exploded[-1]


def explode(node_path):
    """
    Return a list of the node names in the given node path with the node separator removed. For example,
    explode('one:two:three')
    returns
    ['one', 'two', 'three']

    :param node_path: the node path to explode.
    :return: a list of the nodes in the path.
    """
    return node_path.split(NODE_PATH_SEPARATOR)


def validate_node_path(node_path):
    """
    Raise a MeasurementError if the given non-empty node path is not valid; a valid node path is a string consisting of
    valid Python variables separated by colons, like "one:two:three". The empty string '' corresponds to the root path,
    which cannot contain a measurement, so it is not a valid node path.

    :param node_path: The node path to validate.
    :return: None
    """
    for node in explode(node_path):
        if not node:
            raise MeasurementError("Empty node in {}".format(node_path))
    allowed = r'[^\_\:A-Za-z0-9]'
    if re.search(allowed, node_path):
        raise MeasurementError("Invalid character in {}".format(node_path))
