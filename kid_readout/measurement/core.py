"""
This module is the core of the measurement subpackage.

The main ideas are (1) to provide data container classes that define a format and include basic analysis code for the
data, and (2) to separate these data container classes from the format in which the data are stored on disk. These
goals are implemented using a Measurement class, defined in this module, and a few functions.

Measurements can contain other measurements -- for example, a frequency sweep is a collection of streams of
time-ordered data -- so the both the class structure in memory and the hierarchical structure on disk is a standard
tree. Except for the root node, every node in the tree contains a measurement or a sequence of measurements. (The
root node is reserved for metadata.) Every node can thus be reached by a unique node path. For example, if the root
contained only one Sweepstream measurement with the name "sweepstream," the structure would be

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

This hierarchical structure allows measurements that follow the specified format to be saved to disk and re-created
later


A measurement generically contains arrays that contain most of the data and a state dictionary that holds metadata.
Because of restrictions imposed by the libraries used to store data on disk, the dictionary cannot hold arbitrary
values.


To create a new measurement, simply write a subclass of Measurement. Public attributes of the class will be saved

Each Measurement should be self-contained, meaning that it should contain all data and metadata necessary to
analyze and understand it.

"""
import re
import inspect
import importlib
from collections import OrderedDict

CLASS_NAME = '_class'  # This is the string used by writer objects to save class names.

# These names cannot be used for attributes because they are used as part of the public DataFrame interface.
IO_CLASS_NAME = 'io_class'  # This is the fully-qualified name of the io class used to read a measurement from disk.
IO_MODULE = 'io_module'  # This is the full-qualified name of the module used to read and write legacy data.
ROOT_PATH = 'root_path'  # This is the root file or directory from which a measurement was read from disk.
NODE_PATH = 'node_path'  # This is the node path from the root node to the measurement node.

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
    When a measurement is read from disk, the read() function inspects the signature of its __init__ method. The
    advantage of this is that every measurement read from disk can re-initialize itself. However, because only public
    attributes are saved, this creates the restriction that a measurement must save the values with which it is
    initialized and cannot discard them or store them internally under other names. For example, if a measurement's
    __init__() method contained an argument foo that is stored as self.bar, the value in bar will be saved to disk as
    bar but the read() function will not be able to tell that this value should be passed as the value of parameter
    foo. See the instantiate() function.

    Arrays.
    Each measurement has a dimensions class attribute that contains metadata for its array dimensions. This is
    necessary for the netCDF4 io module to handle the array dimensions correctly, and it also allows the classes to
    check the dimensions of their arrays on instantiation through the _validate_dimensions() method. Currently,
    the only array shapes supported are 1-D arrays and N-D arrays for which each dimension corresponds to an existing
    1-D array. The dimensions metadata is implemented as an OrderedDict so that the netCDF4 writer can create the
    dimensions in the proper order. The format is 'array_name': dimension_tuple, where dimension tuple is (
    'array_name') for 1-D arrays or ('some_1D_array', 'another_1D_array'), for example.

    Containers.
    Measurements store state information in a dictionary. (They actually use a subclass called StateDict, which has
    extra access features.) Implementations should be able to store the following objects:
    -basic types, such as numbers, booleans, strings, and None;
    -sequences, i.e. lists, tuples, and arrays, that contain exactly one of the above basic types except None and no containers;
    -dictionaries whose keys are strings and whose values are dictionaries, sequences, or basic types; the contained
     dictionaries and sequences have the same requirements as above.
    Most of these restrictions come from the limitations of netCDF4 and json. Some of them could be relaxed with more
    work.

    Note that NaN == Nan evaluates to False (the same is true of np.nan), and NaN is NaN may evaluate to True or
    False, depending on how the NaN objects are created. The testing function compare_measurements() in
    test/utility.py compares two measurements element-wise, so using NaN in such measurements will cause tests to
    fail even if the measurements are the same.
    """

    dimensions = OrderedDict()

    def __init__(self, state, analyze=False, description='Measurement'):
        """
        Return a new Measurement instance.

        :param state: a dictionary of state information.
        :param analyze: if True (default False), the class will call its analyze() method after instantiation.
        :param description: a string describing the measurement.
        :return: a new Measurement instance.
        """
        self.state = to_state_dict(state)
        self.description = description
        self._parent = None
        self._io_class = None
        self._root_path = None
        self._node_path = None
        self._validate_dimensions()
        if analyze:
            self.analyze()

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
        for name, dimensions in self.dimensions.items():
            if not getattr(self, name).shape == tuple(getattr(self, dimension).size for dimension in dimensions):
                raise ValueError("Shape of {} does not match size of {}.".format(name, dimensions))


class MeasurementSequence(object):
    """
    This is a dummy class used as a marker so that Measurements can contain sequences of other Measurements.
    """
    def __init_(self):
        raise NotImplementedError()

    @property
    def shape(self):
        return (len(self),)


class MeasurementTuple(tuple, MeasurementSequence):
    """
    Measurements containing tuples of Measurements should use instances of this class so that loading and saving are
    handled automatically.
    """
    pass


class MeasurementList(list, MeasurementSequence):
    """
    Measurements containing lists of Measurements should use instances of this class so that loading and saving are
    handled automatically.
    """
    pass


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
    __getattr__ = dict.get
    __delattr__ = dict.__delitem__
    __copy__ = lambda self: StateDict(self)
    __getstate__ = lambda: None
    __slots__ = ()

# TODO: incorporate restrictions into __init__().
# TODO: implement more error checking.
def to_state_dict(dictionary):
    if not all([isinstance(k, (str, unicode)) for k in dictionary]):
        raise MeasurementError("Dictionary keys must be strings.")
    dicts = [(k, v) for k, v in dictionary.items() if isinstance(v, dict)]
    others = [(k, v) for k, v in dictionary.items() if not isinstance(v, dict)]
    return StateDict([(k, v) for k, v in others] +
                     [(k, to_state_dict(v)) for k, v in dicts])


class IO(object):
    """
    This is an abstract class that specifies the IO interface.

    Implementations should be able to store large numpy arrays efficiently. Classes should include the dimensions
    metadata that describes the relationships between the large arrays in the class.
    """

    def __init__(self, root_path):
        """
        Return a new IO object that will read to or write from the given root directory or file. If the root does not
        exist, it should be created. Implementations should NEVER open an existing file in write or append mode, and in
        general should make it impossible to overwrite data.

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

    def measurement_names(self, node_path):
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


# Public functions

def write(measurement, io, node_path):
    """
    Write a measurement to disk using the given IO class. This function feeds data to this class and tells it when to
    create new nodes to match the hierarchy of measurements.

    :param io: an instance of a class that implements the io interface.
    :param node_path: the node_path to the node that will contain this object; all but the final node in node_path must
      already exist.
    :return: None
    """
    _write_node(measurement, io, node_path)

def read(io, node_path, extras=True, translate=None):
    """
    Read a measurement from disk and return it.

    :param io: an instance of a class that implements the IO interface.
    :param node_path:the path to the node to be loaded, in the form 'node0:node1:node2'
    :param extras: add extra variables; see instantiate().
    :param translate: a dictionary with entries 'original_class': 'new_class'; all class names must be fully-qualified.
    :return: the measurement corresponding to the given node.
    """
    if translate is None:
        translate = {}
    measurement = _read_node(io, node_path, extras, translate)
    return measurement


def instantiate(full_class_name, variables, extras):
    """
    Import and instantiate a class using the data in the given dictionary.

    :param full_class_name: the fully-qualified class name as a string, e.g. 'kid_readout.measurement.core.Measurement'.
    :param variables: a dictionary whose keys are the names of the variables available for instantiation and whose
      values are the corresponding values; it must include entries for all non-keyword arguments to __init__().
    :param extras: if True, everything in variables that is not an argument to __init__() will be added as an attribute
      after instantiation; be careful with this.
    :return: an instance of full_class_name instantiated using the given variables.
    """
    class_ = _get_class(full_class_name)
    args, varargs, keywords, defaults = inspect.getargspec(class_.__init__)
    arg_values = []
    for arg, default in zip(reversed(args), reversed(defaults)):
        arg_values.append(variables.get(arg, default))
    for arg in reversed(args[1:-len(defaults)]):  # The first arg is 'self'
        arg_values.append(variables[arg])
    instance = class_(*reversed(arg_values))
    if extras:
        for key, value in variables.items():
            if key not in args:
                setattr(instance, key, value)
    return instance


def instantiate_sequence(full_class_name, contents):
    return _get_class(full_class_name)(contents)


def is_sequence(full_class_name):
    return issubclass(_get_class(full_class_name), MeasurementSequence)


def from_series(series):
    io = _get_class(series.io_class)(series.root_path)
    return read(io, series.node_path)


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


# Private functions


def _write_node(measurement, io, node_path):
    """
    Create a new node at the given node path using the given IO instance and write the measurement data to it.

    :param measurement: a Measurement instance.
    :param io: an instance of a class that implements the IO interface.
    :param node_path: the path of the new node into which the measurement will be written.
    :return: None
    """
    io.create_node(node_path)
    this_class_name = '{}.{}'.format(measurement.__module__, measurement.__class__.__name__)
    io.write_other(node_path, CLASS_NAME, this_class_name)
    items = [(key, value) for key, value in measurement.__dict__.items()
             if not key.startswith('_') and key not in RESERVED_NAMES and key not in measurement.dimensions]
    for key, value in items:
        if isinstance(value, Measurement):
            _write_node(value, io, join(node_path, key))
        elif isinstance(value, MeasurementSequence):
            sequence_node_path = join(node_path, key)
            io.create_node(sequence_node_path)
            sequence_class_name = '{}.{}'.format(value.__module__, value.__class__.__name__)
            io.write_other(sequence_node_path, CLASS_NAME, sequence_class_name)
            for index, meas in enumerate(value):
                _write_node(meas, io, join(sequence_node_path, str(index)))
        else:
            io.write_other(node_path, key, value)
    # Saving arrays in order allows the netCDF group to create the dimensions.
    for array_name, dimensions in measurement.dimensions.items():
        io.write_array(node_path, array_name, getattr(measurement, array_name), dimensions)


def _read_node(io, node_path, extras, translate):
    original_class_name = io.read_other(node_path, CLASS_NAME)
    class_name = translate.get(original_class_name, original_class_name)
    measurement_names = io.measurement_names(node_path)
    if is_sequence(class_name):
        # Use the name of each measurement, which is an int, to restore the order in the sequence.
        contents = [_read_node(io, join(node_path, measurement_name), extras, translate)
                    for measurement_name in sorted(measurement_names, key=int)]
        current = instantiate_sequence(class_name, contents)
    else:
        variables = {}
        for measurement_name in measurement_names:
            variables[measurement_name] = _read_node(io, join(node_path, measurement_name), extras, translate)
        array_names = io.array_names(node_path)
        for array_name in array_names:
            variables[array_name] = io.read_array(node_path, array_name)
        other_names = [vn for vn in io.other_names(node_path)]
        for other_name in other_names:
            variables[other_name] = io.read_other(node_path, other_name)
        current = instantiate(class_name, variables, extras)
    current._io_class = '{}.{}'.format(io.__module__, io.__class__.__name__)
    current._root_path = io.root_path
    current._node_path = node_path
    return current


def _get_class(full_class_name):
    module_name, class_name = full_class_name.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


