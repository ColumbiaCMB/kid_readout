from kid_readout.measurement import core


class Dictionary(core.IO):
    """
    This class implements the IO interface in memory using dictionaries. It is intended for testing.
    """

    _array = '_array'
    _node = '_node'

    def __init__(self, root_path=None, metadata=None):
        """
        Return a new diskless Dictionary IO object.

        :param root_path: the path to the root directory or file; not used.
        :return: a new Dictionary object that can read from and write to the root object at the given path.
        """
        super(Dictionary, self).__init__(root_path=root_path, metadata=metadata)

    def _root_path_exists(self, root_path):
        return root_path is not None

    def _open_existing(self, root_path):
        """
        This is a hacky way of 'opening' an existing root.

        """
        return root_path

    def _create_new(self, root_path):
        return {self._node: {},
                self._array: {}}

    def close(self):
        """
        Close open files.
        """
        self._root = None

    @property
    def closed(self):
        """
        Return True if all files are closed.
        """
        return self._root is None

    def create_node(self, node_path):
        """
        Create a node at the end of the given path; all but the final node in the path must already exist.
        """
        core.validate_node_path(node_path)
        existing, new = core.split(node_path)
        existing_node = self._get_node(existing)
        existing_node[self._node][new] = {self._node: {},
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
        try:
            return node[key]
        except KeyError:
            raise ValueError("Name not found: {}".format(key))

    def measurement_names(self, node_path='/'):
        """
        Return the names of all measurements contained in the measurement at node_path.
        """
        node = self._get_node(node_path)
        return node.get(self._node, {}).keys()

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
        return [k for k in node if not k.startswith('_')]

    # Private methods.

    def _get_node(self, node_path):
        core.validate_node_path(node_path)
        if node_path.startswith(core.NODE_PATH_SEPARATOR):
            node_path = node_path[1:]
        node = self._root
        for name in core.explode(node_path):
            node = node[self._node][name]
        return node
