# -*- coding: utf-8 -*-

"""

@author: bartulem

Convert the labeled tracking .csv to .pkl for loading into the GUI.

The Jython GUI we use to visualize the behavior of the animal requires
the labeled .csv data to have a certain format (in order to be loaded).
This script performs those modifications and when the process is complete,
the .pkl file can be loaded in the *trackedpointdata_V3_5_LEDs.py* version
of the GUI which allows for the visualization of arena LEDs as well.

"""

import os
import sys
import csv
import time
import pickle
import numpy as np


class Transformer:

    # initializer / instance attributes
    def __init__(self, the_csv, sync_pkl):
        self.the_csv = the_csv
        self.sync_pkl = sync_pkl

    def csv_to_pkl(self, **kwargs):

        """
        Inputs
        ----------
        **kwargs: dictionary
        frame_rate : str (file path)
            The empirical camera frame rate for that session; defaults to 120.
        npx_sampling_rate : int/float
            The sampling rate of the NPX system; defaults to 3e4.
        ground_probe : int/float
            In a dual probe setting, the probe the other is synced to; defaults to 0.
        session_timestamps : boolean (0/False or 1/True)
            Whether to take session timestamps (1) for start/stop recording or tracking (0); defaults to 1.
        ----------

        Outputs
        ----------
        final : dictionary
            A dictionary with the tracking data and other information; saved as .pkl file.
        ----------
        """

        # check that all the prerequisites are there
        if not os.path.exists(self.the_csv):
            print('Could not find {}, try again.'.format(self.the_csv))
            sys.exit()

        if not os.path.exists(self.sync_pkl):
            print('Could not find {}, try again.'.format(self.sync_pkl))
            sys.exit()

        print('Working on file: {}.'.format(self.the_csv))
        start_time = time.time()

        # load empirical frame rate from file if it exists
        with open(self.sync_pkl, 'rb') as the_pkl:
            full_pkl_file = pickle.load(the_pkl)
            empirical_frame_rate = full_pkl_file.iloc[0, -1]

        # valid values for booleans
        valid_booleans = [0, False, 1, True]

        frame_rate = float(kwargs['frame_rate'] if 'frame_rate' in kwargs.keys() else 120. if (np.isnan(empirical_frame_rate) or int(round(empirical_frame_rate)) != 120) else empirical_frame_rate)
        npx_sampling_rate = int(kwargs['npx_sampling_rate'] if 'npx_sampling_rate' in kwargs.keys() else 3e4)
        ground_probe = int(kwargs['ground_probe'] if 'ground_probe' in kwargs.keys() else 0)
        session_timestamps = kwargs['session_timestamps'] if 'session_timestamps' in kwargs.keys() and kwargs['session_timestamps'] in valid_booleans else 1

        # get tracking data from .csv
        data = []
        with open(self.the_csv, 'r') as csvfile:
            for rowindx, row in enumerate(csv.reader(csvfile, delimiter=',')):
                if rowindx == 3:
                    labels_raw = row[2:]
                elif rowindx >= 7:
                    data.append(row)

        frames = len(data)

        # get relevant imec ground probe timestamps
        if session_timestamps:
            imec_data_col = full_pkl_file.columns.tolist().index('imec{}'.format(ground_probe))
            session_start = int(full_pkl_file.iloc[0, imec_data_col]) / npx_sampling_rate
            tracking_start = int(full_pkl_file.iloc[1, imec_data_col]) / npx_sampling_rate
            session_end = int(full_pkl_file.iloc[-1, imec_data_col]) / npx_sampling_rate
            tracking_end = int(full_pkl_file.iloc[-2, imec_data_col]) / npx_sampling_rate
        else:
            time_data_col = full_pkl_file.columns.tolist().index('time (ms)')
            session_start = int(full_pkl_file.iloc[1, time_data_col]) / 1e3
            tracking_start = int(full_pkl_file.iloc[1, time_data_col]) / 1e3
            session_end = int(full_pkl_file.iloc[-2, time_data_col]) / 1e3
            tracking_end = int(full_pkl_file.iloc[-2, time_data_col]) / 1e3

        time_stamps = {'startratcamtimestamp': tracking_start,
                       'stopratcamtimestamp': tracking_end,
                       'startsessiontimestamp': session_start,
                       'stopsessiontimestamp': session_end,
                       'starttrackingtimestamp': tracking_start,
                       'stoptrackingtimestamp': tracking_end}

        # load .csv file and get labels data
        labels_dict = {'Marker1': 0, 'Marker2': 1, 'Marker3': 2, 'Marker4': 3,
                       'Neck': 4, 'Back': 5, 'Ass': 6, 'LED1': 7, 'LED2': 8, 'LED3': 9}

        num_of_markers = len(labels_dict.keys())

        labels = []
        for albaelind, alabel in enumerate(labels_raw):
            if albaelind % 3 == 0 and 'LED' not in alabel:
                labels.append(labels_dict[alabel.split(':')[-1]])
            elif albaelind % 3 == 0 and 'LED' in alabel:
                labels.append(labels_dict['LED{}'.format(alabel.split(':')[-1][-1])])

        # get the final output dict ready (num_of_markers is the number of points, 5 is big_x, Y, Z, label, nans)
        final = {'points': [[[] for i in range(5)] for j in range(num_of_markers)],
                 'pointlabels': {0: ['cyan',
                                     'first head'],
                                 1: ['dodger blue',
                                     'second head'],
                                 2: ['lawn green',
                                     'third head'],
                                 3: ['dark magenta',
                                     'fourth head'],
                                 4: ['red',
                                     'neck'],
                                 5: ['yellow',
                                     'middle'],
                                 6: ['green',
                                     'ass'],
                                 7: ['golden rod',
                                     'LED Marker 1'],
                                 8: ['golden rod',
                                     'LED Marker 2'],
                                 9: ['golden rod',
                                     'LED Marker 3']},
                 'boundingboxscaleX': 0,
                 'boundingboxscaleY': 0,
                 'starttrackingtimestamp': time_stamps['starttrackingtimestamp'],
                 'stoptrackingtimestamp': time_stamps['stoptrackingtimestamp'],
                 'startsessiontimestamp': time_stamps['startsessiontimestamp'],
                 'stopsessiontimestamp': time_stamps['stopsessiontimestamp'],
                 'startratcamtimestamp': time_stamps['startratcamtimestamp'],
                 'stopratcamtimestamp': time_stamps['stopratcamtimestamp'],
                 'boundingboxrotation': 0,
                 'headXarray': [],
                 'framerate': frame_rate,
                 'headZarray': [],
                 'headYarray': [],
                 'headoriginarray': [],
                 'boundingboxtransX': 0,
                 'boundingboxtransY': 0}

        # put everything in its place and deal with the nans
        coordinates_order = [2, 0, 1]
        for i in range(num_of_markers):
            for jdx, j in enumerate(coordinates_order):
                big_x = [x[2 + 3 * i + j] for x in data]
                final['points'][i][jdx] = ([float(x) if x else -1000000 for x in big_x])
            final['points'][i][3] = [labels[i]] * frames
            final['points'][i][4] = [0.0] * frames

        for i in range(num_of_markers):
            the_list = final['points'][i][1]
            temp = [ind for ind in range(len(the_list)) if the_list[ind] == -1000000]
            for j in temp:
                final['points'][i][3][j] = -1000000
                final['points'][i][4][j] = -1000000

        # save result to file
        with open('{}.pkl'.format(self.the_csv[:-4]), 'wb') as f:
            # protocol 2 because the GUI requires it / this may change in subsequent renditions
            pickle.dump(final, f, protocol=2)

        print('Conversion complete! The process took {:.2f} seconds.\n'.format(time.time() - start_time))
