# -*- coding: utf-8 -*-

"""

@author: bartulem

Assign real times (based on elapsed IPIs) to spikes.

If there is sync misalignment between tracking and imec clocks,
such that sampling frequencies cannot be trusted, it is possible
to convert every event into time/frame measured by the IPI generator.

That works in the following way:
(1) spikes that happen before or after tracking are eliminated
(2) each spike time is calculated as the sum between how much time
    elapsed until the LED just prior to the spike, and the sampling
    rate between two bounding LEDs multiplied by the number of samples
    until the spike

!NB: Spike times are now zeroed to tracking start!

"""

import numpy as np
from numba import njit


@njit(parallel=False)
def times_to_events(sync_data,
                    event_data,
                    imec_data_col,
                    time_data_col,
                    frame_data_col):

    """
    Inputs
    ----------
    sync_data : np.ndarray
        The sync data; necessary input.
    event_data : np.ndarray
        The spike train; necessary input.
    imec_data_col : int
        IMEC column in the sync DataFrame; necessary input.
    time_data_col : int
        IPI time column in the sync DataFrame; necessary input.
    frame_data_col : int
        Tracking column in the sync DataFrame; necessary input.
    ----------

    Outputs
    ----------
    new_event_times : np.ndarray
        An array with spike times converted to the synce pulse clock.
    new_event_frames : np.ndarray
        An array with spike frames adjusted to the synce pulse clock.
    ----------
    """

    # get first and last imec LEDs
    first_led_samples = sync_data[1, imec_data_col]
    last_led_samples = sync_data[-2, imec_data_col]

    # eliminate events before 1st and after last LED onset
    # this means there are no more spikes before or after tracking
    event_data = event_data[(event_data > first_led_samples) & (event_data < last_led_samples)]

    # create new array for event times
    new_event_times = np.zeros(len(event_data))
    new_event_frames = np.zeros(len(event_data), dtype=np.int64)

    for idx, spike in enumerate(event_data):

        for xx in range(1, sync_data.shape[0] - 2):

            # find bounding LED events in samples
            lower_bound_samples = sync_data[xx, imec_data_col]
            upper_bound_samples = sync_data[xx + 1, imec_data_col]

            if lower_bound_samples <= spike < upper_bound_samples:

                # find bounding LED events in sync time and frames
                lower_bound_time = sync_data[xx, time_data_col]
                upper_bound_time = sync_data[xx + 1, time_data_col]

                lower_bound_frames = sync_data[xx, frame_data_col]
                upper_bound_frames = sync_data[xx + 1, frame_data_col]

                # get total number of samples between two bounding LED events
                total_samples_between = upper_bound_samples - lower_bound_samples

                # samples between spike and lower bound led
                samples_to_spike = spike - lower_bound_samples + 1

                # get total time/frames between two bounding LED events
                total_time_between = upper_bound_time - lower_bound_time
                total_frames_between = upper_bound_frames - lower_bound_frames

                # calculate spike time/frame
                spike_time = lower_bound_time + (total_time_between / total_samples_between) * samples_to_spike
                spike_frame = (lower_bound_frames - sync_data[1, frame_data_col]) + \
                              int(round((total_frames_between / total_samples_between) * samples_to_spike))

                # save results converted to seconds/frames to array
                new_event_times[idx] = spike_time / 1e3
                new_event_frames[idx] = spike_frame

                break

    return new_event_times, new_event_frames
