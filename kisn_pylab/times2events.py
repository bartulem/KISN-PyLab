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


class AssignTimes:

    # initializer / instance attributes
    def __init__(self, sync_data, event_data):
        self.sync_data = sync_data
        self.event_data = event_data

    def times_to_events(self, **kwargs):

        """
        Inputs
        ----------
        **kwargs: dictionary
        probe_id : int
            The ID of the imec probe the event_data was recorded on; defaults to 0.
        ----------

        Outputs
        ----------
        new_event_times : np.ndarray
            An array with spike times converted to the synce pulse clock.
        new_event_frames : np.ndarray
            An array with spike frames adjusted to the synce pulse clock.
        ----------
        """

        probe_id = int(kwargs['probe_id'] if 'probe_id' in kwargs.keys() else 0)

        # convert event_data to array form
        if type(self.event_data) != np.ndarray:
            self.event_data = np.array(self.event_data)

        # find data columns
        imec_data_col = self.sync_data.columns.tolist().index('imec{}'.format(probe_id))
        time_data_col = self.sync_data.columns.tolist().index('time (ms)')
        frame_data_col = self.sync_data.columns.tolist().index('tracking')

        # get first and last imec LEDs
        first_led_samples = self.sync_data.iloc[1, imec_data_col]
        last_led_samples = self.sync_data.iloc[-2, imec_data_col]

        # eliminate events before 1st and after last LED onset
        # this means there are no more spikes before or after tracking
        self.event_data = self.event_data[(self.event_data > first_led_samples)
                                          & (self.event_data < last_led_samples)]

        # create new array for event times/frames
        new_event_times = np.zeros(len(self.event_data))
        new_event_frames = np.zeros(len(self.event_data))

        for idx, spike in enumerate(self.event_data):
            truth = True
            while truth:
                for xx in range(1, self.sync_data.shape[0] - 2):

                    # find bounding LED events in samples
                    lower_bound_samples = self.sync_data.iloc[xx, imec_data_col]
                    upper_bound_samples = self.sync_data.iloc[xx + 1, imec_data_col]

                    if lower_bound_samples <= spike < upper_bound_samples:

                        # find bounding LED events in sync time and frames
                        lower_bound_time = self.sync_data.iloc[xx, time_data_col]
                        upper_bound_time = self.sync_data.iloc[xx + 1, time_data_col]

                        lower_bound_frames = self.sync_data.iloc[xx, frame_data_col]
                        upper_bound_frames = self.sync_data.iloc[xx + 1, frame_data_col]

                        # get total number of samples between two bounding LED events
                        total_samples_between = upper_bound_samples - lower_bound_samples

                        # samples between spike and lower bound led
                        samples_to_spike = spike - lower_bound_samples + 1

                        # get total time/frames between two bounding LED events
                        total_time_between = upper_bound_time - lower_bound_time
                        total_frames_between = upper_bound_frames - lower_bound_frames

                        # calculate spike time/frame
                        spike_time = lower_bound_time + (total_time_between / total_samples_between) * samples_to_spike
                        spike_frame = (lower_bound_frames - self.sync_data.iloc[1, frame_data_col]) + \
                                      int(round((total_frames_between / total_samples_between) * samples_to_spike))

                        # save results converted to seconds/frames to array
                        new_event_times[idx] = spike_time / 1e3
                        new_event_frames[idx] = spike_frame

                        truth = False

        return new_event_times, new_event_frames
