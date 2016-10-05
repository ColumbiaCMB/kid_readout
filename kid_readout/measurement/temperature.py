import numpy as np

from kid_readout.measurement import core
from kid_readout.analysis.resources import experiments


def add_temperature(measurement, cryostat, recursive=True, overwrite=False):
    start_epoch = measurement.start_epoch()
    if cryostat.lower() == 'hpd':
        from kid_readout.equipment import hpd_temps as temps
        thermometry = None
    elif cryostat.lower() == 'starcryo':
        from kid_readout.equipment import starcryo_temps as temps
        thermometry = experiments.get_experiment_info_at(start_epoch, cryostat)['thermometry_config']
    else:
        raise ValueError("Invalid cryostat: {}".format(cryostat))
    _add_temperature(measurement, temps, thermometry, recursive=recursive, overwrite=overwrite)


def _add_temperature(measurement, temps, thermometry, recursive, overwrite):
    if 'temperature' in measurement.state and not overwrite:
        raise ValueError("{} already has {}".format(measurement, measurement.state.temperature))
    if hasattr(measurement, 'epoch'):
        measurement.state.temperature = valid_temperatures(measurement.epoch, temps, thermometry)
    else:
        start_epoch = measurement.start_epoch()
        if not np.isnan(start_epoch):
            measurement.state.temperature = valid_temperatures(start_epoch, temps, thermometry)
    if recursive:
        for key, value in measurement.__dict__.items():
            if not key.startswith('_'):
                if isinstance(value, core.Measurement):
                    _add_temperature(value, temps, thermometry, recursive=recursive, overwrite=overwrite)
                elif isinstance(value, core.MeasurementList):
                    for measurement in value:
                        _add_temperature(measurement, temps, thermometry, recursive=recursive, overwrite=overwrite)


def valid_temperatures(epoch, temps, thermometry):
    # We could use thermometry to get the names that are used there, but it's not clear to me if we want to do this.
    names = ('package', 'package_secondary', 'load', 'aux')
    return core.StateDict([(name, temp) for name, temp in zip(names, temps.get_temperatures_at(epoch))
                           if not np.isnan(temp)])

