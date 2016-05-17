"""
The main goals of this subpackage are (1) to provide data container classes that define a format and include basic
analysis code for the data, and (2) to separate these data container classes from the format in which the data are
stored on disk. These goals are implemented using a Measurement class and a few functions and auxiliary classes.

The format is designed to handle hierarchical measurements naturally. For example, a frequency sweep is a collection
of streams of time-ordered data, so the both the class structure in memory and the hierarchical structure on disk are
a standard tree. This structure allows measurements that follow the specified format to be saved to disk and
re-created later without any additional metadata.

The IO abstract class in this module controls reading to and writing from disk. Implementations of this class must
implement its abstract methods using some data format.

To create a new measurement, simply write a subclass of Measurement that follows these rules:
  - All required arguments to the __init__() method are stored in the class as public attributes with the same name.
  - Any arguments that are sequences of measurements use the MeasurementList class to contain them.
  - All numpy ndarrays have an entry in the dimensions OrderedDict that specifies their size, possibly relative to other
    arrays in the class; see the Measurement docstring.
  - The __init__() method should call Measurement.__init__() after performing all other setup (otherwise,
    Measurement._validate_dimensions() could fail to find the arrays that it needs.)
  - All other public attributes obey the restrictions described in the Measurement docstring.
  - None of the entries in RESERVED_NAMES are used as public attributes.

The arguments to __init__() define the data format. Example:

# The class must inherit Measurement.
class TimeOrderedData(Measurement):

    # These entries assert that the class has public attribute 'time' that is a 1-D numpy ndarray, and that it has
    # a public attribute 'data' that is also a numpy ndarray with data.shape = (time.size,), i.e. the arrays have
    # the same shape.
    dimensions = OrderedDict([('time', ('time',)), ('data', ('time',))])

    def __init__(self, data, time, measurement, meas_list, not_an_array, analyze_me=False, state=None, description=''):
        # 1-D arrays data and time have to be public attributes and must have entries in dimensions.
        self.data = data
        self.time = time
        # This other Measurement subclass will be automatically saved and loaded.
        self.measurement = measurement
        # This MeasurementList and its contents will also be automatically saved and loaded.
        self.meas_list = meas_list
        # This has no default value so it also has to be saved. It has no entry in dimensions so it should not be an
        # array. See the Measurement docstring for restrictions on its value.
        self.not_an_array = not_an_array
        # Call Measurement.__init__() after assigning all arrays.
        super(TimeOrderedData, self).__init__(state=state, description=description)
        # This doesn't have to be recorded as an attribute because it has a default value, so the measurement can be
        # resurrected without a value for it saved on disk.
        if analyze_me:
            self.analyze_me()

The data in this class will be saved to disk by the IO.write() function and re-instantiated by the IO.read() function.
If a class contains attributes that are either measurements or sequences of measurements (that use the MeasurementList
container), these will also be saved and restored correctly. Each Measurement should be self-contained, meaning that it
should contain all data and metadata necessary to analyze and understand it.

Because the same measurement class can potentially describe many measurements that use different equipment and thus
have different state data, the best way to handle presence or absence of equipment or settings is simply through the
presence or absence of entries in the state dictionary. It's easy for external code to check
if 'some_equipment_info' in state: analyze_it()
to determine whether to analyze the information. If an entry has to be present for some reason, using None is better
than NaN to signify "not available," even if the state value is usually a number. An example of a good reason would
be if some equipment was used during the measurement, but its state was not available when the measurement was
written. Besides the advantage of avoiding comparison subtleties, None will cause numeric operations to blow up
immediately.

An instantiated IO class creates a root node that can contain multiple measurements. Except for the root node, every
node in the tree contains a measurement or a sequence of measurements. (The root node is reserved for metadata.) Every
node can thus be reached by a unique node path starting from the root, denoted by a slash.

For example, if the root contained two SweepStream measurements, the structure could be

/SweepStream0
  /stream
  /sweep
    /streams
      /0
      /1
      etc.
/SweepStream1
etc.

The node path to the first SweepStream would be just "/SweepStream0" while the node path to one of the streams in its
sweep could be "/SweepStream0/sweep/streams/1", and so on. A valid node path is a string consisting of valid node names
separated by slashes. A valid node name is either a valid Python variable or a string representing a nonnegative
integer. Nodes that are Python variables correspond to attributes of a Measurement subclass or the measurements saved at
the root level, while nodes that are integers correspond to an index in a MeasurementList.

The IO class uses the node paths to save and load the data structure without knowledge of the underlying format. For
example, if io is a subclass of IO, then io.read('SweepStream0') would read the entire first SweepStream from disk,
while io.read('SweepStream0/stream') would read just its Stream measurement. (These node paths could also be specified
as absolute paths, starting with a slash, but the IO class accepts either because it becomes the top-level node for
the trees that it reads and writes.)
"""
__author__ = 'flanigan'
