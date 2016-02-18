import os
import json
import numpy as np
from kid_readout.measure import measurement


def new(location, name):
    new_location = os.path.join(location, name)
    os.mkdir(new_location)
    return new_location


def write(thing, location, name):
    if isinstance(thing, np.ndarray):
        np.save(os.path.join(location, name + '.npy'), thing)
    else:
        with open(os.path.join(location, name), 'w') as f:
            json.dump(thing, f)


def read(top, memmap=False):
    if memmap:
        mmap_mode = 'r'
    else:
        mmap_mode = None
    return _visit(None, top, mmap_mode)


def _visit(parent, location, mmap_mode):
    names = [f for f in os.listdir(location) if os.path.isdir(os.path.join(location, f))]
    with open(os.path.join(location, measurement.CLASS_NAME)) as f:
        class_name = json.load(f)
    class_ = measurement.get_class(class_name)
    if measurement.is_sequence(class_):
        current = class_([_visit(None, os.path.join(location, name), mmap_mode)
                          for name in sorted(names, key=int)])
        for meas in current:
            meas._parent = parent  # Preserve the current convention.
    else:
        current = class_()
        for name in names:
            setattr(current, name, _visit(current, os.path.join(location, name), mmap_mode))
    current._parent = parent
    # Load files
    filenames = [filename for filename in os.listdir(location)
                 if filename not in measurement.RESERVED_NAMES and os.path.isfile(os.path.join(location, filename))]
    for filename in filenames:
        full = os.path.join(location, filename)
        name, extension = os.path.splitext(filename)
        if extension == '.npy':
            setattr(current, name, np.load(full, mmap_mode=mmap_mode))
        else:
            with open(full, 'r') as f:
                setattr(current, name, json.load(f))
    return current
