"""
This is a plotting module for classes in basic.py.
"""
from __future__ import division
from collections import namedtuple

import numpy as np


sweep_raw_defaults = {'linestyle': 'none',
                      'marker': ',',
                      'color': 'black',
                      'alpha': 0.2}

sweep_mean_defaults = {'linestyle': 'none',
                       'marker': '.',
                       'markersize': 2,
                       'color': 'blue',
                       'alpha': 1}

stream_raw_defaults = {'linestyle': 'none',
                       'marker': ',',
                       'color': 'green',
                       'alpha': 0.2}

model_defaults = {'linestyle': '-',
                  'linewidth': 0.3,
                  'color': 'brown',
                  'alpha': 1}

resonance_defaults = {'linestyle': 'none',
                      'marker': '.',
                      'markersize': 2,
                      'color': 'brown',
                      'alpha': 1}


def resonator_complex_plane(resonator, axis, normalize=True, num_model_points=1000,
                            sweep_mean_settings=None, model_settings=None, resonance_settings=None):
    sweep_mean_kwds = sweep_mean_defaults.copy()
    if sweep_mean_settings is not None:
        sweep_mean_kwds.update(sweep_mean_settings)
    model_kwds = model_defaults.copy()
    if model_settings is not None:
        model_kwds.update(model_settings)
    resonance_kwds = resonance_defaults.copy()
    if resonance_settings is not None:
        resonance_kwds.update(resonance_settings)
    rd = resonator.extract(normalize=normalize, num_model_points=num_model_points)
    axis.plot(rd.s21_data.real, rd.s21_data.imag, **sweep_mean_kwds)
    axis.plot(rd.s21_model.real, rd.s21_model.imag, **model_kwds)
    axis.plot(rd.s21_0.real, rd.s21_0.imag, **resonance_kwds)
    return rd


def sss_complex_plane(sss, axis, normalize=True, num_model_points=1000, zoom=False,
                      sweep_raw_settings=None, sweep_mean_settings=None, stream_raw_settings=None, model_settings=None,
                      resonance_settings=None):
    sweep_raw_kwds = sweep_raw_defaults.copy()
    if sweep_raw_settings is not None:
        sweep_raw_kwds.update(sweep_raw_settings)
    stream_raw_kwds = stream_raw_defaults.copy()
    if stream_raw_settings is not None:
        stream_raw_kwds.update(stream_raw_settings)
    for stream in sss.sweep.streams:
        if normalize:
            s21 = sss.resonator.remove_background(frequency=stream.frequency, s21_raw=stream.s21_raw)
        else:
            s21 = stream.s21_raw
        axis.plot(s21.real, s21.imag, **sweep_raw_kwds)
    if normalize:
        stream_s21 = sss.resonator.remove_background(frequency=sss.stream.frequency, s21_raw=sss.stream.s21_raw)
    else:
        stream_s21 = sss.stream.s21_raw
    axis.plot(stream_s21.real, stream_s21.imag, **stream_raw_kwds)
    rd = resonator_complex_plane(resonator=sss.resonator, axis=axis, normalize=normalize,
                                 num_model_points=num_model_points, sweep_mean_settings=sweep_mean_settings,
                                 model_settings=model_settings, resonance_settings=resonance_settings)
    if zoom:
        x_span = np.abs(rd.s21_0.real - stream_s21.real.mean())
        y_span = np.abs(rd.s21_0.imag - stream_s21.imag.mean())
        span = np.max([x_span, y_span, 0.1])
        axis.set_xlim(rd.s21_0.real - span, rd.s21_0.real + span)
        axis.set_ylim(rd.s21_0.imag - span, rd.s21_0.imag + span)
