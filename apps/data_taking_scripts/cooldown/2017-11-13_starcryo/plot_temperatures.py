import time
from collections import deque

import matplotlib.pyplot as plt
from matplotlib import animation

from kid_readout.equipment import starcryo_temps


def animate(framedata, times, data, axes, lines):
    times.append(framedata[0])
    for point, queue in zip(framedata[1:], data):
        queue.append(point)
    t_padding = 0.1
    for ax, queue, line in zip(axes, data, lines):
        ax.set_xlim(times[0] - t_padding, times[-1] + t_padding)
        y_padding = 0.1 * (max(queue) - min(queue)) or 1
        ax.set_ylim(min(queue) - y_padding, max(queue) + y_padding)
        line.set_xdata(times)
        line.set_ydata(queue)
    axes[0].set_xticklabels([])
    axes[1].set_xticklabels([])
    for ax in axes:
        ax.set_yticklabels(['{:.3f}'.format(tick) for tick in ax.get_yticks()])
    return axes, lines


def get_temperatures():
    t0 = time.time()
    while True:
        temps = starcryo_temps.get_current_temperatures()
        yield (temps['unix_time'] - t0, temps['eccosorb_diode_temperature'], temps['stepper_diode_temperature'],
               temps['package_ruox4550_temperature'])


def main():
    frame_delay_ms = 1000  # in ms
    length = 128  # the number of samples displayed
    eccosorb_data = deque([], maxlen=length)
    stepper_data = deque([], maxlen=length)
    package_data = deque([], maxlen=length)
    time_data = deque([], maxlen=length)
    fig, (eccosorb_ax, stepper_ax, package_ax ) = plt.subplots(3, 1, figsize=(3, 3))
    eccosorb_ax.set_ylabel('Eccosorb / K', fontsize='xx-small')
    stepper_ax.set_ylabel('stepper / K', fontsize='xx-small')
    package_ax.set_ylabel('package / K', fontsize='xx-small')
    package_ax.set_xlabel('time / s', fontsize='xx-small')
    eccosorb_line, = eccosorb_ax.plot(time_data, eccosorb_data, '-r')
    stepper_line, = stepper_ax.plot(time_data, stepper_data, '-r')
    package_line, = package_ax.plot(time_data, package_data, '-r')
    _ = animation.FuncAnimation(fig, animate, frames=get_temperatures,
                                fargs=(time_data, (eccosorb_data, stepper_data, package_data),
                                       (eccosorb_ax, stepper_ax, package_ax),
                                       (eccosorb_line, stepper_line, package_line)),
                                interval=frame_delay_ms)
    plt.show()


if __name__ == '__main__':
    main()

