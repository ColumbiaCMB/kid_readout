from kid_readout.measurement import core


class Dictionary(core.IO):
    """
    This class implements the IO interface in memory using dictionaries. It is intended for testing.
    """

    _array = '_array'
    _measurement = '_measurement'

    def __init__(self, root_path=None):
        """
        Return a new diskless Dictionary object.

        :param root_path: the path to the root directory or file; not used.
        :return: a new Dictionary object that can read from and write to the root object at the given path.
        """
        super(Dictionary, self).__init__(root_path=root_path)
        self.root = {self._measurement: {},
                     self._array: {}}

    def close(self):
        """
        Close open files.
        """
        self.root = None

    @property
    def closed(self):
        """
        Return True if all files are closed.
        """
        return self.root is None

    def create_node(self, node_path):
        """
        Create a node at the end of the given path; all but the final node in the path must already exist.
        """
        core.validate_node_path(node_path)
        existing, new = core.split(node_path)
        existing_node = self._get_node(existing)
        existing_node[self._measurement][new] = {self._measurement: {},
                                                 self._array: {}}

    def write_other(self, node_path, key, value):
        """
        Write value to node_path with name key.
        """
        node = self._get_node(node_path)
        node[key] = value

    def write_array(self, node_path, key, value, dimensions):
        """
        Write value, a numpy array, to node_path with name key.
        """
        node = self._get_node(node_path)
        node[self._array][key] = value

    def read_array(self, node_path, key):
        """
        Read array key from node_path.
        """
        node = self._get_node(node_path)
        return node[self._array][key]

    def read_other(self, node_path, key):
        """
        Read non-array object with name key from node_path.
        """
        node = self._get_node(node_path)
        return node[key]

    def measurement_names(self, node_path='/'):
        """
        Return the names of all measurements contained in the measurement at node_path.
        """
        node = self._get_node(node_path)
        return node.get(self._measurement, {}).keys()

    def array_names(self, node_path):
        """
        Return the names of all arrays contained in the measurement at node_path.
        """
        node = self._get_node(node_path)
        return node.get(self._array, {}).keys()

    def other_names(self, node_path):
        """
        Return the names of all other variables contained in the measurement at node_path.
        """
        node = self._get_node(node_path)
        return [k for k in node if k not in core.RESERVED_NAMES and not k.startswith('_')]

    # Private methods.

    def _get_node(self, node_path):
        core.validate_node_path(node_path)
        if node_path.startswith(core.NODE_PATH_SEPARATOR):
            node_path = node_path[1:]
        node = self.root
        for name in core.explode(node_path):
            node = node[self._measurement][name]
        return node
