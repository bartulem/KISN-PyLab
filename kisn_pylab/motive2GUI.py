# -*- coding: utf-8 -*-

"""

@author: bartulem

Convert the labeled tracking .csv to .pkl for loading into the GUI.

The Jython GUI we use to visualize the behavior of the animal requires
the labeled .csv data to have a certain format (in order to be loadable).
This script performs those modifications and when the process is complete,
the .pkl file can be loaded in the *trackedpointdata_V3_5_LEDs.py* version
of the GUI which allows for the visualization of arena LEDs as well.

"""

import pandas as pd
import os
import sys
import csv
import time
import pickle


class MotiveGUI:

    # initializer / instance attributes
    def __init__(self, the_csv, the_pkl):
        self.the_csv = the_csv
        self.the_pkl = the_pkl

    def csv_to_pkl(self, **kwargs):

        """
        Parameters
        ----------
        **kwargs: dictionary
        frame_rate : str (file path)
            The empirical camera frame rate for that session; defaults to 120.
        npx_sampling_rate : int/float
            The sampling rate of the NPX system; defaults to 3e4.
        reference_probe : int/float
            The probe other probes are referenced to; defaults to 0 (as in imec0).
        ----------
        """

        # test that all the prerequisites are there
        if not os.path.exists(self.the_csv):
            print('Could not find {}, try again.'.format(self.the_csv))
            sys.exit()

        if not os.path.exists(self.the_pkl):
            print('Could not find {}, try again.'.format(self.the_pkl))
            sys.exit()

        print('Working on file: {}.'.format(self.the_csv))
        t = time.time()

        # load empirical frame rate from file if it exists
        with open(self.the_pkl, 'rb') as the_pkl:
            full_pkl_file = pickle.load(the_pkl)
            empirical_frame_rate = full_pkl_file.iloc[0, -1]

        frame_rate = float([kwargs['frame_rate'] if 'frame_rate' in kwargs.keys() else 120. if int(round(empirical_frame_rate)) != 120 else empirical_frame_rate][0])
        npx_sampling_rate = int([kwargs['npx_sampling_rate'] if 'npx_sampling_rate' in kwargs.keys() else 3e4][0])
        reference_probe = int([kwargs['reference_probe'] if 'reference_probe' in kwargs.keys() else 0][0])

        # get tracking data from .csv
        data = []
        with open(self.the_csv, 'r') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            next(spamreader, 7)
            next(spamreader, 7)
            next(spamreader, 7)
            next(spamreader, 7)
            next(spamreader, 7)
            next(spamreader, 7)
            next(spamreader, 7)

            for row in spamreader:
                data.append(row)

        frames = len(data)

        # load .txt file and get relevant timestamps
        time_stamps = {'startratcamtimestamp': 0, 'stopratcamtimestamp': 0, 'startsessiontimestamp': 0, 'stopsessiontimestamp': int(full_pkl_file.iloc[-1, reference_probe]),
                       'starttrackingtimestamp': int(full_pkl_file.iloc[2, reference_probe]), 'stoptrackingtimestamp': int(full_pkl_file.iloc[-3, reference_probe])}

        # load .csv file and get labels data
        labels_dict = {'Marker1': 0, 'Marker2': 1, 'Marker3': 2, 'Marker4': 3, 'Neck': 4, 'Back': 5, 'Ass': 6, 'LED1': 7, 'LED2': 8, 'LED3': 9}
        num_of_markers = len(labels_dict.keys())
        labels_data = pd.read_csv(self.the_csv, sep=',', nrows=2)
        labels_raw = labels_data.iloc[1, :].tolist()[1:]
        labels = []
        for albaelind, alabel in enumerate(labels_raw):
            if albaelind % 3 == 0 and 'LED' not in alabel:
                labels.append(labels_dict[alabel.split(':')[-1]])
            elif albaelind % 3 == 0 and 'LED' in alabel:
                labels.append(labels_dict['LED{}'.format(alabel.split(':')[-1][-1])])

        # get the final output dict ready (num_of_markers is the number of points, 5 is X, Y, Z, label, nans)
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
                 'starttrackingtimestamp': time_stamps['starttrackingtimestamp'] / npx_sampling_rate,
                 'stoptrackingtimestamp': time_stamps['stoptrackingtimestamp'] / npx_sampling_rate,
                 'startsessiontimestamp': time_stamps['startsessiontimestamp'] / npx_sampling_rate,
                 'stopsessiontimestamp': time_stamps['stopsessiontimestamp'] / npx_sampling_rate,
                 'startratcamtimestamp': time_stamps['startratcamtimestamp'] / npx_sampling_rate,
                 'stopratcamtimestamp': time_stamps['stopratcamtimestamp'] / npx_sampling_rate,
                 'boundingboxrotation': 0,
                 'headXarray': [],
                 'frame_rate': frame_rate,
                 'headZarray': [],
                 'headYarray': [],
                 'headoriginarray': [],
                 'boundingboxtransX': 0,
                 'boundingboxtransY': 0}

        # put everything in its place and deal with the nans
        coordinates_order = [0, 2, 1]
        for i in range(num_of_markers):
            tt = 0
            for j in coordinates_order:
                X = [x[2 + 3 * i + j] for x in data]
                final['points'][i][tt] = ([(float(x) if x else -1000000) for x in X])
                tt = tt + 1
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
            # protocol 2 because the GUI doesn't recognize anything above
            pickle.dump(final, f, protocol=2)

        print('Conversion complete! The process took {:.2f} seconds.'.format(time.time() - t))
