"""
This module implements reading and writing of Measurements using a directory hierarchy and numpy arrays.

Each node is a string representing a directory.
Numpy arrays are stored as .npy files.
Other attributes are stored using json.

Limitations:
Because json has only a single sequence type, all sequences that are not numpy arrays are returned as lists.
"""
import os
import json
import numpy as np
from kid_readout.measurement import core


class IO(core.IO):

    def __init__(self, root_path):
        self.root_path = os.path.expanduser(root_path)
        if not os.path.isdir(self.root_path):
            os.mkdir(root_path)
        self.root = self.root_path

    def close(self):
        self.root = None

    def create_node(self, node_path):
        os.mkdir(os.path.join(self.root, *core.explode(node_path)))

    def write_array(self, node_path, key, value, dimensions):
        node = self._get_node(node_path)
        np.save(os.path.join(node, key + '.npy'), value)

    def write_other(self, node_path, key, value):
        node = self._get_node(node_path)
        with open(os.path.join(node, key), 'w') as f:
            json.dump(value, f)

    def read_array(self, node_path, name, memmap=False):
        if memmap:
            mmap_mode = 'r'
        else:
            mmap_mode = None
        full = os.path.join(self._get_node(node_path), name + '.npy')
        return np.load(full, mmap_mode=mmap_mode)

    def read_other(self, node_path, name):
        with open(os.path.join(self._get_node(node_path), name)) as f:
            return json.load(f)

    def get_measurement_names(self, node_path):
        node = self._get_node(node_path)
        return [f for f in os.listdir(node)if os.path.isdir(os.path.join(node, f))]

    def get_array_names(self, node_path):
        node = self._get_node(node_path)
        return [os.path.splitext(f)[0] for f in os.listdir(node) if os.path.isfile(os.path.join(node, f))
                and os.path.splitext(f)[1] == '.npy']

    def get_other_names(self, node_path):
        node = self._get_node(node_path)
        return [f for f in os.listdir(node) if os.path.isfile(os.path.join(node, f))
                and not f in core.RESERVED_NAMES and os.path.splitext(f)[1] != '.npy']

    def _get_node(self, node_path):
        full_path = os.path.join(self.root, *core.explode(node_path))
        if not os.path.isdir(full_path):
            raise ValueError("Invalid path: {}".format(full_path))
        return full_path

