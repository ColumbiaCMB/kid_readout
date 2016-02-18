import os
from inspect import isclass
import importlib

CLASS_NAME = '_class_name'  # This is the string used by writer objects to save class names.
RESERVED_NAMES = [CLASS_NAME]

# Import all classes from the subpackage "measurements"; this function is called at the end of the module.
_measurements = {}
def _load_measurements():
    subpackage_path = os.path.join(os.path.dirname(__file__), 'measurements')
    filenames = [fn for fn in os.listdir(subpackage_path) if not fn.startswith('_') and fn.endswith('.py')]
    for filename in filenames:
        module_name, extension = os.path.splitext(filename)
        module = importlib.import_module('.' + module_name, 'kid_readout.measurement.measurements')
        _measurements.update((name, class_) for name, class_ in module.__dict__.items()
                             if isclass(class_) and issubclass(class_, (Measurement, MeasurementSequence)))


def get_class(class_name):
    """
    Return a Measurement class by name.

    :param class_name: the name of the class as a string.
    :return: the corresponding class from any module in the measurements subpackage.
    """
    return _measurements[class_name]


def is_sequence(class_):
    return issubclass(class_, MeasurementSequence)


class Measurement(object):
    """
    This is an abstract class that represents a measurement for a single channel.

    Measurements are hierarchical: a Measurement can contain other Measurements.

    Each Measurement should be self-contained, meaning that it should contain all data and metadata necessary to
    analyze and understand it. Can this include temperature data?

    Caching: all raw data attributes are public and all special or processed data attributes are private.
    """

    def __init__(self, state=None, analyze=False):
        self._parent = None
        if state is None:
            self.state = {}
        else:
            self.state = state
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
        :return: a DataFrame containing all of the instance attributes.
        """
        pass

    def write(self, writer, location, name):
        """
        Write this measurement to disk using the given writer object. The abstraction used here is that a location is
        a container for hierarchically-organized data.

        :param writer: an object that implements the writer interface.
        :param location: the existing root location into which this object will be written.
        :param name: the name of the location containing this object.
        :return: None

        For example, when called with parameters (writer, 'root', 'measurement'), the location "root" must already
        exist and the data from this measurement will be stored within the location root/measurement
        """
        self_location = writer.new(location, name)
        writer.write(self.__class__.__name__, self_location, CLASS_NAME)
        for name, thing in self.__dict__.items():
            if not name.startswith('_'):
                if isinstance(thing, Measurement):
                    thing.write(writer, self_location, name)
                elif isinstance(thing, MeasurementSequence):
                    sequence_location = writer.new(self_location, name)
                    writer.write(thing.__class__.__name__, sequence_location, CLASS_NAME)
                    for index, meas in enumerate(thing):
                        meas.write(writer, sequence_location, str(index))
                else:
                    writer.write(thing, self_location, name)
        return self_location


class MeasurementSequence(object):
    """
    This is a dummy class that exists so that Measurements can contain sequences of other Measurements.
    """
    pass


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


# TODO: Gross, but true -- this code currently has to be below the definition of the Measurement class in this file.
_load_measurements()