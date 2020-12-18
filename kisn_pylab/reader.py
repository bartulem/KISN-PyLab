# -*- coding: utf-8 -*-

"""

@author: bartulem

Read in sync events from all desired data streams.

Neuropixel recordings are saved into a *1D binary vector*, from a 2D array organised as 385 rows (channels) x n columns (samples).
The data are written to file from this matrix in column-major (F) order, i.e., the first sample in the recording was written to file
for every channel, then the second sample was written for every channel, etc. Neuropixel raw data is provided as an int16 binary.
Neuropixel ADCs are 10 bits, with a range of -0.6 to 0.6 V, and acquisition was at 500x gain, yielding a resolution of 2.34 µV/bit.
To obtain readings in µV, you should multiply the int16 values by 2.34.

This script has the purpose of extracting LEDon sync events from four independent data streams: (1) NPX recording files (may be one or
two in a given session, depending on the number of probes), (2) the exported Motive tracking file, and (3) the IMU sensor file,
(4) the sync IPI generator, all of which keep track of the LEDon occurrences. Unlike v2.0.0, future versions will assume the LED
pulses are generated continuously and randomly (by default with a fixed 250ms duration, and 250-1500ms IPIs).

The code first goes through all of those files and picks up the LED events, (taking into account that the LED signal jitters,
which can be probe independent). It then selects a portion of the starting and ending LED pulses in the tracking data stream
(10 by default) and tries to match these templates in the other data streams, aligning the start and end of tracking to different
recordings. All the corresponding LED events in all data streams are stored in a DataFrame, which is saved to a separate
binary .pkl file. This code might consume a lot of RAM (64Gb should be enough in most cases).

"""

import os
import sys
import numpy as np
import gc
import time
import csv
import pandas as pd
import pickle
from collections import Counter
from tqdm.notebook import tqdm
import operator
import warnings
warnings.simplefilter('ignore')
from astropy.convolution import convolve
from astropy.convolution import Gaussian1DKernel


class EventReader:

    # initializer / instance attributes
    def __init__(self, npx_files, sync_df):
        self.npx_files = npx_files
        self.sync_df = sync_df

    def read_se(self, **kwargs):

        """
        Inputs
        ----------
        **kwargs: dictionary
        nchan : int/float
            Total number of channels on the NPX probe, for probe 3B2 should be 385; defaults to 385.
        sync_chan : int/float
            Sync events channel number, for probe 3B2 should be 385; defaults to 385.
        track_file : str/boolean
            The absolute path to the tracking data .csv file; defaults to 0.
        imu_file : str/boolean
            The absolute path to the IMU data .txt file; defaults to 0.
        sync_led_duration : int/float
            The duration of ON time for sync LEDs; defaults to 250 (ms).
        sync_led_error : int/float
            The possible error in the duration of ON time for sync LEDs; defaults to 50 (ms).
        ground_probe : int
            In a dual probe setting, the probe the other is synced to; defaults to 0.
        frame_rate : int/float
            The tracking camera frame rate for that session; defaults to 120.
        npx_sampling_rate : int/float
            The sampling rate of the NPX system; defaults to 3e4.
        sync_sequence : int/float
            The length of the sequence the LED events should be matched across data streams; defaults to 10.
        sample_error : int/float
            The time the presumed IMEC/IMU LEDs could be allowed to err around; defaults to 30 (ms).
        which_imu_time : int/float
            The IMU time to be used in the analyses, loop.starttime (0) or sample.time (1); defaults to 1.
        sync_ipi_file : str/boolean
            The absolute path to the sync IPI data .txt file; defaults to 0.
        ----------

        Outputs
        ----------
        track_file : csv object
            The corrected and LED-shortened tracking data; saved as a separate .csv file.
        imu_file : pd.DataFrame
            The IMU data; saved as .pkl file.
        sync_df : pd.DataFrame
            The sync data; saved as .pkl file.
        ----------
        """

        nchan = int(kwargs['nchan'] if 'nchan' in kwargs.keys() and (type(kwargs['nchan']) == int or type(kwargs['nchan']) == float) else 385)
        sync_chan = int(kwargs['sync_chan'] if 'sync_chan' in kwargs.keys() and (type(kwargs['sync_chan']) == int or type(kwargs['sync_chan']) == float) else 385)
        track_file = kwargs['track_file'] if 'track_file' in kwargs.keys() and type(kwargs['track_file']) == str else 'eldiablomuerte'
        imu_file = kwargs['imu_file'] if 'imu_file' in kwargs.keys() and type(kwargs['imu_file']) == str else 'eldiablomuerte'
        sync_led_duration = int(kwargs['sync_led_duration'] if 'sync_led_duration' in kwargs.keys() and (type(kwargs['sync_led_duration']) == int or type(kwargs['sync_led_duration']) == float) else 250)
        sync_led_error = int(kwargs['sync_led_error'] if 'sync_led_error' in kwargs.keys() and (type(kwargs['sync_led_error']) == int or type(kwargs['sync_led_error']) == float) else 50)
        ground_probe = int(kwargs['ground_probe'] if 'ground_probe' in kwargs.keys() else 0)
        frame_rate = float(kwargs['frame_rate'] if 'frame_rate' in kwargs.keys() else 120.)
        npx_sampling_rate = int(kwargs['npx_sampling_rate'] if 'npx_sampling_rate' in kwargs.keys() else 3e4)
        sync_sequence = int(kwargs['sync_sequence'] if 'sync_sequence' in kwargs.keys() else 10)
        sample_error = int(kwargs['sample_error'] if 'sample_error' in kwargs.keys() and (type(kwargs['sample_error']) == int or type(kwargs['sample_error']) == float) else 30)
        which_imu_time = int(kwargs['which_imu_time'] if 'which_imu_time' in kwargs.keys() and (type(kwargs['which_imu_time']) == int or type(kwargs['which_imu_time']) == float) else 1)
        sync_ipi_file = kwargs['sync_ipi_file'] if 'sync_ipi_file' in kwargs.keys() and type(kwargs['sync_ipi_file']) == str else 'eldiablomuerte'

        # check that the NPX files are there
        for anpxfile in self.npx_files:
            if not os.path.exists(anpxfile):
                print('Could not find file {}, try again.'.format(anpxfile))
                sys.exit()

        start_time = time.time()
        print('Extracting sync data from NPX file(s), please be patient - this could take >5 minutes.\n')

        # initialize dictionary to store the data in
        sync_dict = {}

        # extract LED events from the NPX file(s)
        for anpxfile in self.npx_files:

            # memory maps are used for accessing small segments of large files on disk, without reading the entire file into memory.
            npx_recording = np.memmap(anpxfile, mode='r', dtype=np.int16, order='C')

            # integer divide the length of the recording by channel count to get number of samples
            npx_samples = npx_recording.shape[0] // nchan

            # reshape the array such that channels are rows and samples are columns
            npx_recording = npx_recording.reshape((nchan, npx_samples), order='F')

            # find sync events and collect them in a dict / this warrants a more detailed explanation
            # the sync channel consists of 16 1-bit channels, and when its values are read as np.int16 they can assume
            # different values such as 0, 64, 128, 132, etc. (to be converted back to binary, use np.binary_repr(num, 16)).
            # The sync LED event should be a sequence of 64s that is 250 ms in duration (this is the default value that
            # can be set as an input parameter), a LED off period should be an, at least equal, sequence of zeros.
            # The other values occur mainly because of noise. So, in theory the LED events should last 7500 samples
            # (250ms * 30) but for different reasons may last less and might include jitter values (128, 132, etc.),
            # so in screening for LED events that is taken into account.

            counter = sorted(dict(Counter(npx_recording[sync_chan - 1, :])).items(), key=operator.itemgetter(1), reverse=True)
            most_freq_two_values = [counter[0][0], counter[1][0]]

            session_proportion = (counter[0][1] + counter[1][1]) / npx_samples
            if session_proportion < .99:
                print('The two most dominant values, {} and {} appear together only {:.3f} of the total session, '
                      'so something is wrong. Check it out!'.format(counter[0][0], counter[1][0], session_proportion))
                sys.exit()

            high_val = np.nanmax(most_freq_two_values)
            low_val = np.nanmin(most_freq_two_values)

            # how many samples to screen LED events for
            imec_check_samples = 30 * (sync_led_duration - sync_led_error)

            probe_sync = []
            counter_on = 0
            probe_sync.append(0)
            for inxSync, itemSync in enumerate(tqdm(npx_recording[sync_chan - 1, :])):
                if 0 < inxSync < npx_samples - imec_check_samples \
                        and itemSync == high_val and npx_recording[sync_chan - 1, inxSync - 1] == low_val \
                        and (npx_recording[sync_chan - 1, max(0, inxSync - imec_check_samples):inxSync] != high_val).all() \
                        and (npx_recording[sync_chan - 1, (inxSync + 1):(inxSync + imec_check_samples + 1)] != low_val).all():
                    probe_sync.append(inxSync)
                    counter_on += 1
            probe_sync.append(npx_samples)

            # delete the map object from memory
            del npx_recording
            gc.collect()

            # save data to sync_dict where imec number is key
            sync_dict['imec{}'.format(anpxfile[anpxfile.find('imec') + len('imec')])] = probe_sync
            print('There are {} total LED events in the imec{} file.'.format(counter_on, anpxfile[anpxfile.find('imec') + len('imec')]))

        # extract LED events from the tracking file
        if os.path.exists(track_file):

            print('Extracting sync data from the tracking file, please remain patient.')

            # read in the 4th line of the tracking .csv and find the column where LED appears for the first time
            with open(track_file, 'r') as thecsv:
                for rowindx, row in enumerate(csv.reader(thecsv, delimiter=',')):
                    if rowindx == 3:
                        columnofint = next(indx for indx, obj in enumerate(row) if 'LED' in obj)
                        break

            # correct fully empty frames / this warrants a more detailed explanation
            # tracking files, for some reason, sometimes get frames where all markers drop out. This happens
            # infrequently and when it does it is only a couple of frames that are surrounded by frames that have
            # all markers present. This can be a problem for LED detection because one empty frame would separate
            # something that was a unitary LED event into two different LED events. There's also the problem
            # where only the LED rigid body markers get lost for a couple of frames. To prevent this staying in
            # the data, the next snippet of code goes through the tracking file and interpolates those empty
            # frames by smoothing over a window of the LED duration (250 ms or 30 rows above and below the frame).

            # the duration of sync LEDs in tracking frames (rounded to the nearest decade)
            tracking_check_samples = int(round(sync_led_duration / (1 / frame_rate * 1000), -1))

            # the half of tracking check samples
            track_smooth_window = int(round(tracking_check_samples / 2))

            # define led columns
            led_cols = list(range(columnofint, columnofint + 9))

            # load original tracking data
            original_tracking_data = pd.read_csv(track_file, sep=',', skiprows=6)
            total_frame_number = original_tracking_data.shape[0]

            print('Correcting empty frames.\n')

            corrected_frames = {}
            for i in tqdm(range(total_frame_number - 1)):

                # The following conditions are remedied by this snippet of code >

                # (1) values are missing for all animal points outside of a LED event
                statement_1 = (original_tracking_data.iloc[i, 2:].isnull().values.all() and
                               original_tracking_data.iloc[max(0, i - track_smooth_window):i, columnofint:(columnofint + 9)].isnull().values.all() and
                               original_tracking_data.iloc[(i + 1):min(i + track_smooth_window + 1, total_frame_number - 1), columnofint:(columnofint + 9)].isnull().values.all())

                # (2) values are missing for all points during a LED event
                statement_2 = (original_tracking_data.iloc[i, 2:].isnull().values.all() and
                               original_tracking_data.iloc[max(0, i - track_smooth_window):i, columnofint:(columnofint + 9)].notnull().values.any() and
                               original_tracking_data.iloc[(i + 1):min(i + track_smooth_window + 1, total_frame_number - 1), columnofint:(columnofint + 9)].notnull().values.any())

                # (3) values are missing for just the LED points (during a LED event)
                statement_3 = (original_tracking_data.iloc[i, 2:].notnull().values.any() and
                               original_tracking_data.iloc[i, columnofint:(columnofint + 9)].isnull().values.all() and
                               original_tracking_data.iloc[max(0, i - track_smooth_window):i, columnofint:(columnofint + 9)].notnull().values.any() and
                               original_tracking_data.iloc[(i + 1):min(i + track_smooth_window + 1, total_frame_number - 1), columnofint:(columnofint + 9)].notnull().values.any())

                if statement_1 or statement_2 or statement_3:
                    # print('Frame {} is being corrected.'.format(original_tracking_data.iloc[i, 0]))
                    changed_array = np.zeros(original_tracking_data.shape[1])
                    for inx in range(len(changed_array)):
                        if inx < 2:
                            if inx < 1:
                                changed_array[inx] = original_tracking_data.iloc[i, inx]
                            else:
                                changed_array[inx] = round(original_tracking_data.iloc[i, inx], 6)
                        elif 1 < inx and inx not in led_cols:
                            if statement_1 or statement_2:
                                changed_array[inx] = round(convolve(original_tracking_data.iloc[max(0, i - track_smooth_window):min(i + track_smooth_window + 1, total_frame_number - 1), inx],
                                                                    kernel=Gaussian1DKernel(stddev=2), nan_treatment='interpolate', preserve_nan=False)[track_smooth_window], 6)
                            else:
                                changed_array[inx] = round(original_tracking_data.iloc[i, inx], 6)
                        else:
                            if statement_1:
                                changed_array[inx] = np.nan
                            else:
                                changed_array[inx] = round(convolve(original_tracking_data.iloc[max(0, i - track_smooth_window):min(i + track_smooth_window + 1, total_frame_number - 1), inx],
                                                                    kernel=Gaussian1DKernel(stddev=2), nan_treatment='interpolate', preserve_nan=False)[track_smooth_window], 6)

                    # format strings accordingly
                    changed_list = []
                    for idx, item in enumerate(changed_array):
                        if idx == 0:
                            changed_list.append(str(int(item)))
                        else:
                            if not np.isnan(item):
                                changed_list.append('{:.6f}'.format(item))
                            else:
                                changed_list.append('')
                    corrected_frames[i + 7] = changed_list

            # save the interpolated tracking .csv
            with open(track_file, 'r') as rtf, open('{}_interpolated.csv'.format(track_file[:-4]), 'w', newline='') as wtf:
                writer = csv.writer(wtf)
                reader = csv.reader(rtf)
                for rowindx, row in enumerate(reader):
                    if rowindx not in corrected_frames.keys():
                        writer.writerow(row)
                    else:
                        writer.writerow(corrected_frames[rowindx])

            # load the df and search for LEDons/offs
            tracking_data = pd.read_csv('{}_interpolated.csv'.format(track_file[:-4]), sep=',', skiprows=6)

            tracking_sync = {}
            tracking_on = 1
            all_led_frames = []

            print('Looking for LED events in the corrected tracking file.\n')

            for row in tqdm(range(tracking_data.shape[0])):
                if row > 0 and not tracking_data.iloc[row, columnofint:(columnofint + 9)].isnull().values.all() \
                        and tracking_data.iloc[max(row - tracking_check_samples, 0):row, columnofint:(columnofint + 9)].isnull().values.all() \
                        and tracking_data.iloc[row:min(row + tracking_check_samples, tracking_data.shape[0]), columnofint:(columnofint + 9)].isnull().all(axis=1).sum() == 0:
                    tracking_sync['{}LEDon'.format(tracking_on)] = tracking_data.loc[row, 'Frame']
                    all_led_frames.append(tracking_data.loc[row, 'Frame'])
                    tracking_on += 1

            # save data to sync_dict
            sync_dict['tracking'] = tracking_sync
            print('Completed! There are {} total LED events in the tracking file.'.format(len(tracking_sync.keys())))

            # save the tracking .csv with everything before the first LEDon and after the last LEDon removed
            with open('{}_interpolated.csv'.format(track_file[:-4]), 'r') as inp, \
                    open('{}final.csv'.format('{}_interpolated.csv'.format(track_file[:-4])[:-16]), 'w', newline='') as out:
                writer = csv.writer(out)
                for rowindx, row in enumerate(csv.reader(inp)):
                    if rowindx < 7 or (all_led_frames[0] + 7) <= rowindx <= (all_led_frames[-1] + 7):
                        writer.writerow(row)

            # remove interpolated file from the tracking directory
            os.remove('{}_interpolated.csv'.format(track_file[:-4]))

        else:
            print('Tracking file not given or found.')

        # extract LED events from the IMU file
        if os.path.exists(imu_file):

            print('Extracting sync data from the IMU file, please remain patient.\n')

            # load the files as df
            imu_df = pd.read_csv(imu_file, sep=',', header=None)

            # get rid of the date in the first column (loop.starttime)
            imu_df.iloc[:, 0] = [np.int64(x.split('\t')[1]) for x in imu_df.iloc[:, 0]]

            # add the imu header
            imu_df.columns = ['loop.starttime (ms)', 'sample.time (ms)', 'acc.x', 'acc.y', 'acc.z',
                              'linacc.x', 'linacc.y', 'linacc.z', 'gyr.x', 'gyr.y', 'gyr.z',
                              'mag.x', 'mag.y', 'mag.z', 'euler.x', 'euler.y', 'euler.z',
                              'LED', 'sound', 'sys.cal', 'gyr.cal', 'acc.cal', 'mag.cal']

            # get LED times
            sample_array = imu_df.iloc[:, which_imu_time].tolist()
            led_array = imu_df['LED'].tolist()
            teensy_time = []
            imu_led = []
            imu_on = 0
            for idx, item in enumerate(tqdm(led_array)):
                if idx > 0 and item != 0 and led_array[idx - 1] == 0:
                    imu_led.append(idx)
                    teensy_time.append(sample_array[idx])
                    imu_on += 1

            # save data to sync_dict
            sync_dict['imu_frame_number'] = imu_led
            sync_dict['teensy_sample_time'] = teensy_time
            print('Completed! here are {} total LED events in the IMU file.'.format(imu_on))

            # save IMU data to file
            with open('{}.pkl'.format(imu_file[:-4]), 'wb') as imu_data:
                pickle.dump(imu_df, imu_data)

        else:
            print('IMU file not given or found.')

        # align LED events from the tracking data stream to other data streams
        if os.path.exists(track_file):

            # find the matching sequence of IPIs in the IPI .txt file
            if os.path.exists(sync_ipi_file):

                print('Template matching IPIs to tracking data.')

                # load IPI file and clean up the first column
                ipi_df = pd.read_csv(sync_ipi_file, sep=',', header=None)
                ipi_df.iloc[:, 0] = [np.int64(x.split('\t')[1]) for x in ipi_df.iloc[:, 0]]

                # add sync_led_duration to every IPI
                ipis = ipi_df.to_numpy().ravel() + sync_led_duration

                # find sequence in tracking data
                first_diffs = (np.diff(list(sync_dict['tracking'].values())[:sync_sequence]) + 1) * (1e3 / frame_rate)
                last_diffs = (np.diff(list(sync_dict['tracking'].values())[-sync_sequence:]) + 1) * (1e3 / frame_rate)

                start_end_ipis = []

                for idx, item in enumerate(ipis):
                    if idx < len(ipis) - sync_sequence:
                        temp_diffs = ipis[idx:idx + sync_sequence - 1]
                        if (np.absolute(temp_diffs - first_diffs) <= sample_error).all():
                            print('Found first matching IPI as {} ms.'.format(item))
                            start_end_ipis.append(idx)
                        elif (np.absolute(temp_diffs - last_diffs) <= sample_error).all():
                            print('Found last matching IPI as {} ms.'.format(ipis[idx + sync_sequence - 2]))
                            start_end_ipis.append(idx + sync_sequence - 2)
                            break

                if len(start_end_ipis) == 0 or start_end_ipis[0] == start_end_ipis[-1]:
                    print('At least one IPI not matched, check it out!')
                    sys.exit()

                # convert IPIs to time elapsed since first pulse
                tracking_ipis = ipis[start_end_ipis[0]:(start_end_ipis[-1] + 1)]

                ipis_time = np.zeros(len(tracking_ipis) + 1)
                for idx in range(len(tracking_ipis)):
                    ipis_time[idx + 1] = np.sum(tracking_ipis[:(idx + 1)])

                # save data to sync_dict
                ipis_dict = {}
                counter_on = 1
                for idx, led_time in enumerate(ipis_time):
                    ipis_dict['{}LEDon'.format(counter_on)] = led_time
                    counter_on += 1

                sync_dict['time (ms)'] = ipis_dict

                print('After template matching, there are now {} clock times based on IPIs.'.format(len(ipis_time)))

            for data_key in sync_dict.keys():
                if 'imec' in data_key:
                    print('Template matching tracking to {} LED events.'.format(data_key))

                    if os.path.exists(sync_ipi_file):
                        first_diffs = tracking_ipis[:(sync_sequence - 1)] * 30
                        last_diffs = tracking_ipis[(-sync_sequence + 1):] * 30
                    else:
                        first_diffs = (np.diff(list(sync_dict['tracking'].values())[:sync_sequence]) + 1) * (npx_sampling_rate / frame_rate)
                        last_diffs = (np.diff(list(sync_dict['tracking'].values())[-sync_sequence:]) + 1) * (npx_sampling_rate / frame_rate)

                    imec_leds = sync_dict[data_key][1:-1]
                    important_led_positions = []

                    for idx, item in enumerate(imec_leds):
                        if idx < len(imec_leds) - sync_sequence:
                            temp_diffs = (np.diff(imec_leds[idx:(idx + sync_sequence)]) + 1)
                            if (np.absolute(temp_diffs - first_diffs) <= 30 * sample_error).all():
                                print('Found first matching LED at sample number {}.'.format(item))
                                important_led_positions.append(idx)
                            elif (np.absolute(temp_diffs - last_diffs) <= 30 * sample_error).all():
                                print('Found last matching LED at sample number {}.'.format(imec_leds[idx + sync_sequence - 1]))
                                important_led_positions.append(idx + sync_sequence - 1)
                                break

                    if len(important_led_positions) == 0 or important_led_positions[0] == important_led_positions[-1]:
                        print('At least one matching LED not found, check it out!')
                        sys.exit()

                    imec_leds = imec_leds[important_led_positions[0]:important_led_positions[-1] + 1]
                    print('After template matching, there are now {} LED events in the {} file.'.format(len(imec_leds), data_key))

                    imec_dict = {}
                    counter_on = 1
                    imec_dict['Session start'] = 0
                    for led_event in imec_leds:
                        imec_dict['{}LEDon'.format(counter_on)] = led_event
                        counter_on += 1
                    imec_dict['Session stop'] = sync_dict[data_key][-1]
                    sync_dict[data_key] = imec_dict

                elif 'teensy' in data_key:
                    print('Template matching tracking to {} LED events.'.format(data_key))

                    if os.path.exists(sync_ipi_file):
                        first_diffs = tracking_ipis[:(sync_sequence - 1)]
                        last_diffs = tracking_ipis[(-sync_sequence + 1):]
                    else:
                        first_diffs = (np.diff(list(sync_dict['tracking'].values())[:sync_sequence]) + 1) * (1e3 / frame_rate)
                        last_diffs = (np.diff(list(sync_dict['tracking'].values())[-sync_sequence:]) + 1) * (1e3 / frame_rate)

                    imu_leds = sync_dict[data_key]
                    important_led_positions = []

                    for idx, item in enumerate(imu_leds):
                        if idx < len(imu_leds) - sync_sequence:
                            temp_diffs = (np.diff(imu_leds[idx:idx + sync_sequence]) + 1)
                            if (np.absolute(temp_diffs - first_diffs) <= sample_error).all():
                                print('Found first matching LED at sample time {}.'.format(item))
                                important_led_positions.append(idx)
                            elif (np.absolute(temp_diffs - last_diffs) <= sample_error).all():
                                print('Found last matching LED at sample time {}.'.format(imu_leds[idx + sync_sequence - 1]))
                                important_led_positions.append(idx + sync_sequence - 1)
                                break

                    if len(important_led_positions) == 0 or important_led_positions[0] == important_led_positions[-1]:
                        print('At least one matching LED not found, check it out!')
                        sys.exit()

                    imu_leds = imu_leds[important_led_positions[0]:important_led_positions[-1] + 1]
                    imu_frame_numbers = sync_dict['imu_frame_number'][important_led_positions[0]:important_led_positions[-1] + 1]
                    print('After template matching, there are now {} LED events in the {} file.'.format(len(imu_leds), data_key))

                    teensy_dict = {}
                    imu_frame_dict = {}
                    counter_on = 1
                    for idx, led_event in enumerate(imu_leds):
                        teensy_dict['{}LEDon'.format(counter_on)] = led_event
                        imu_frame_dict['{}LEDon'.format(counter_on)] = imu_frame_numbers[idx]
                        counter_on += 1
                    sync_dict['imu_frame_number'] = imu_frame_dict
                    sync_dict['teensy_sample_time'] = teensy_dict

            # pack sync_dict into a df and save it to a .pkl file
            choose_imec = 'imec{}'.format(ground_probe)
            export_sync_df = pd.DataFrame(index=sync_dict[choose_imec].keys(), columns=sync_dict.keys())
            for data_key in sync_dict.keys():
                for event_key in sync_dict[choose_imec].keys():
                    if event_key in sync_dict[data_key].keys():
                        export_sync_df.loc[event_key, data_key] = sync_dict[data_key][event_key]

            # check if DataFrame contains NANs
            if export_sync_df.iloc[1:-1, :].isnull().values.any():
                print('There is at least one NAN value as a LED event in the saved sync_df! Check it out!')

            with open('{}'.format(self.sync_df), 'wb') as df:
                pickle.dump(export_sync_df, df)

        print('Extraction complete! It took {:.2f} minutes.\n'.format((time.time() - start_time) / 60))
