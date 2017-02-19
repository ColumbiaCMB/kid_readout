"""
This module implements reading and writing of Measurements using a directory hierarchy, numpy arrays, and JSON.

Each node is a directory:
Numpy arrays are stored as .npy files;
Other values are stored using json.

Limitations and issues.

Because json has only a single sequence type, all sequences that are not numpy arrays are returned as lists.
"""
import os
import json

import numpy as np

from kid_readout.measurement import core


# TODO: implement tagging and closing of memmapped files
# TODO: check node path validation -- how were tests passing?
class NumpyDirectory(core.IO):

    # This can be used as a conventional extension for directories created by this IO class, but it is not used or
    # enforced anywhere internally.
    EXTENSION = '.npd'

    def __init__(self, root_path, metadata=None, memmap=False):
        super(NumpyDirectory, self).__init__(root_path=os.path.abspath(os.path.expanduser(root_path)),
                                             metadata=metadata)
        if memmap:
            self._mmap_mode = 'r'
        else:
            self._mmap_mode = None

    def _root_path_exists(self, root_path):
        return os.path.isdir(root_path)

    def _open_existing(self, root_path):
        return root_path

    def _create_new(self, root_path):
        os.mkdir(root_path)
        return root_path

    def close(self):
        """
        Disable further reading or writing of files. Note that this doesn't actually close any memmapped files. The
        numpy.memmap documentation says that to close them you have to delete the memmap object, and it's not clear
        to me how to do that here.
        """
        self._root = None

    @property
    def closed(self):
        return self._root is None

    def create_node(self, node_path):
        existing, new = core.split(node_path)
        if not new:
            raise core.MeasurementError("Cannot create root node.")
        os.mkdir(os.path.join(self._get_node(existing), new))

    def write_array(self, node_path, key, value, dimensions):
        node = self._get_node(node_path)
        filename = os.path.join(node, key + '.npy')
        with self._safe_open(filename) as f:
            np.save(f, value)

    def write_other(self, node_path, key, value):
        node = self._get_node(node_path)
        filename = os.path.join(node, key)
        with self._safe_open(filename) as f:
            try:
                json.dump(value, f)
            except TypeError as e:
                raise ValueError("json.dump({}) of {} ({}) failed: {}".format(key, value, repr(value), e.message))

    def read_array(self, node_path, name):
        full = os.path.join(self._get_node(node_path), name + '.npy')
        return np.load(full, mmap_mode=self._mmap_mode)

    def read_other(self, node_path, name):
        full_name = os.path.join(self._get_node(node_path), name)
        if not os.path.isfile(full_name):
            raise ValueError("Name not found: {}".format(name))
        with open(full_name) as f:
            return json.load(f)

    def node_names(self, node_path='/'):
        node = self._get_node(node_path)
        return [f for f in os.listdir(node) if os.path.isdir(os.path.join(node, f))]

    def array_names(self, node_path):
        node = self._get_node(node_path)
        return [os.path.splitext(f)[0] for f in os.listdir(node) if os.path.isfile(os.path.join(node, f))
                and os.path.splitext(f)[1] == '.npy']

    def other_names(self, node_path):
        node = self._get_node(node_path)
        return [f for f in os.listdir(node)
                if os.path.isfile(os.path.join(node, f)) and
                not f.startswith('_') and
                os.path.splitext(f)[1] != '.npy']

    def _get_node(self, node_path):
        if self.closed:
            raise ValueError("I/O operation on closed file")
        if node_path != '':
            core.validate_node_path(node_path)
        full_path = os.path.join(self._root, *core.explode(node_path))
        if not os.path.isdir(full_path):
            raise ValueError("Invalid path: {}".format(full_path))
        return full_path

    @staticmethod
    def _safe_open(filename):
        if os.path.exists(filename):
            raise RuntimeError("File already exists: {}".format(filename))
        return open(filename, 'w')
