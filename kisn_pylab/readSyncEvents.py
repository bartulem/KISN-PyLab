# -*- coding: utf-8 -*-

"""

@author: bartulem

Read in sync events from all desired data streams.

Neuropixel recordings are saved into a *1D binary vector*, from a 2D array organised as 385 rows (channels) x n columns (samples). 
The data are written to file from this matrix in column-major (F) order, i.e., the first sample in the recording was written to file 
for every channel, then the second sample was written for every channel, etc. Neuropixel raw data is provided as an int16 binary.
Neuropixel ADCs are 10 bits, with a range of -0.6 to 0.6 V, and acquisition was at 500x gain, yielding a resolution of 2.34 µV/bit.
To obtain readings in µV, you should multiply the int16 values by 2.34.

This script has the purpose of extracting sync events from three independent data streams: (1) NPX recording files (may be one or
two in a given session, depending on the number of probes), (2) the exported Motive tracking file, and (3) the IMU sensor file,
all of which keep track of the LEDon & LEDoff occurrences. The code goes through all of those files and picks up the LED events,
saving the complete data frame to a separate binary .pkl file.

"""

import os
import sys
import numpy as np
from tqdm import tqdm
import gc
import time
import csv
import pandas as pd
import pickle


class Sync:

    # initializer / instance attributes
    def __init__(self, npx_files, sync_df):
        self.npx_files = npx_files
        self.sync_df = sync_df

    def read_se(self, **kwargs):

        """
        Parameters
        ----------
        **kwargs: dictionary
        nchan : int/float
            Total number of channels on the NPX probe, for Probe3b should be 385; defaults to 385.
        sync_chan : int/float
            Sync port channel number, for Probe3b should be 385; defaults to 385.
        ledoff : boolean (0/False or 1/True)
            Whether to consider the LEDoff times; defaults to 0.
        track_file : str/boolean
            The path to the tracking data .csv file; defaults to 0.
        imu_file : str/boolean
            The path to the IMU data .txt file; defaults to 0.
        imu_pkl : str/boolean
            The path to the IMU data .pkl file; defaults to 0.
        ----------
        """

        # valid values for booleans
        valid_bools = [0, False, 1, True]

        nchan = int([kwargs['nchan'] if 'nchan' in kwargs.keys() and type(kwargs['nchan']) == int or type(kwargs['nchan']) == float else 385][0])
        sync_chan = int([kwargs['sync_chan'] if 'sync_chan' in kwargs.keys() and type(kwargs['sync_chan']) == int or type(kwargs['sync_chan']) == float else 385][0])
        ledoff = [kwargs['ledoff'] if 'ledoff' in kwargs.keys() and kwargs['ledoff'] in valid_bools else 0][0]
        track_file = [kwargs['track_file'] if 'track_file' in kwargs.keys() and type(kwargs['track_file']) == str else 0][0]
        imu_file = [kwargs['imu_file'] if 'imu_file' in kwargs.keys() and type(kwargs['imu_file']) == str else 0][0]
        imu_pkl = [kwargs['imu_pkl'] if 'imu_pkl' in kwargs.keys() and type(kwargs['imu_pkl']) == str else 0][0]

        # test that the NPX files are there
        for anpxfile in self.npx_files:
            if not os.path.exists(anpxfile):
                print('Could not find file {}, try again.'.format(anpxfile))
                sys.exit()

        t = time.time()
        print('Extracting sync data from NPX files, please be patient - this can take up to several minutes.')

        # initialize dictionary to store the data in
        sync_dict = {}

        # extract LED events from the NPX file(s)
        for anpxfile in self.npx_files:

            # give it a 2s break
            time.sleep(2)

            # memmaps are used for accessing small segments of large files on disk, without reading the entire file into memory.
            npx_recording = np.memmap(anpxfile, mode='r', dtype=np.int16, order='C')

            # integer divide the length of the recording by channel count to get number of samples
            npx_samples = len(npx_recording) // nchan

            # reshape the array such that channels are rows and samples are columns
            npx_recording = npx_recording.reshape((nchan, npx_samples), order='F')

            # get the sync data in a separate array, this is a burden on the system memory
            sync_data = npx_recording[sync_chan - 1, :]

            # find sync events and collect them in a dict
            changepoints = []
            for inxSync, itemSync in tqdm(enumerate(sync_data)):
                if len(changepoints) == 0:
                    if itemSync != 0 and sync_data[inxSync - 1] == 0:
                        changepoints.append(inxSync)
                else:
                    if itemSync != 0 and sync_data[inxSync - 1] == 0:
                        changepoints.append(inxSync - 1)
                    elif itemSync == 0 and sync_data[inxSync - 1] != 0:
                        changepoints.append(inxSync)
            changepoints[-1] = changepoints[-1] - 1

            probe_sync = {}
            counter_on = 1
            counter_off = 1
            probe_sync['Session start'] = 0
            for indx, timestamp in enumerate(changepoints):
                if indx == 0:
                    probe_sync['TTL input start'] = timestamp
                elif indx != 0 and indx != (len(changepoints) - 1) and indx % 2 != 0:
                    probe_sync['{}LEDon'.format(counter_on)] = timestamp
                    counter_on += 1
                elif indx != 0 and indx != (len(changepoints) - 1) and indx % 2 == 0:
                    if ledoff:
                        probe_sync['{}LEDoff'.format(counter_off)] = timestamp
                        counter_off += 1
                    else:
                        continue
                else:
                    probe_sync['TTL input stop'] = timestamp
            probe_sync['Session stop'] = len(sync_data)

            # delete the memmap obj from memory
            del npx_recording
            gc.collect()

            # save data to sync_dict where imec number is key
            sync_dict['imec{}'.format(anpxfile[anpxfile.find('imec') + len('imec')])] = probe_sync

        # extract LED events from the tracking file
        if type(track_file) == 'str' and os.path.exists(track_file):

            print('Extracting sync data from the tracking file, please remain patient.')

            # read in the 4th line of the tracking .csv and find the column where LED appears for the first time
            with open(track_file, 'r') as thecsv:
                for rowindx, row in enumerate(csv.reader(thecsv, delimiter=',')):
                    if rowindx == 3:
                        columnofint = next(indx for indx, obj in enumerate(row) if 'LED' in obj)

            # load the df and search for LEDons/offs
            tracking_data = pd.read_csv(track_file, sep=',', skiprows=6)

            tracking_sync = {}
            tracking_on = 1
            tracking_off = 1
            all_led_frames = []
            tracking_sync['TTL input start'] = 0
            for row in tqdm(range(tracking_data.shape[0])):
                if not np.isnan(tracking_data.iloc[row, columnofint]) and np.isnan(tracking_data.iloc[row - 1, columnofint]):
                    tracking_sync['{}LEDon'.format(tracking_on)] = tracking_data.loc[row, 'Frame']
                    all_led_frames.append(tracking_data.loc[row, 'Frame'])
                    tracking_on += 1
                elif np.isnan(tracking_data.iloc[row, columnofint]) and not np.isnan(tracking_data.iloc[row - 1, columnofint]):
                    if ledoff:
                        tracking_sync['{}LEDoff'.format(tracking_off)] = tracking_data.loc[row, 'Frame']
                        all_led_frames.append(tracking_data.loc[row, 'Frame'])
                        tracking_off += 1
                    else:
                        continue
            tracking_sync['TTL input stop'] = tracking_data.iloc[-1, 0]

            # save data to sync_dict
            sync_dict['tracking'] = tracking_sync

            # save the tracking .csv with everything before the first LEDon and after the last LEDon removed
            with open(track_file, 'r') as inp, open('{}_modified.csv'.format(track_file[:-4]), 'w', newline='') as out:
                writer = csv.writer(out)
                for rowindx, row in enumerate(csv.reader(inp)):
                    if rowindx < 7 or (all_led_frames[0]+7) <= rowindx <= (all_led_frames[-1]+7):
                        writer.writerow(row)

        else:
            print('Tracking file not given or found.')

        # extract LED events from the IMU file
        if type(imu_file) == 'str' and os.path.exists(imu_file):

            print('Extracting sync data from the IMU file, please remain patient.')

            # load the files as df, add the imu header and get LED times
            imu_df = pd.read_csv('{}'.format(imu_file), sep=',', header=None)
            imu_df.columns = ['loop.starttime (ms)', 'sample.time (ms)', 'acc.x', 'acc.y', 'acc.z', 'gyr.x', 'gyr.y', 'gyr.z', 'mag.x', 'mag.y', 'mag.z',
                              'euler.x', 'euler.y', 'euler.z', 'LED', 'sound', 'sys.cal', 'gyr.cal', 'acc.cal', 'mag.cal']

            # save IMU data to file
            if type(imu_pkl) == 'str':
                with open(imu_pkl, 'wb') as imu_data:
                    pickle.dump(imu_df, imu_data)
                print('IMU .pkl file saved at: {}.'.format(imu_pkl))

            sample_array = imu_df['sample.time (ms)'].tolist()
            led_array = imu_df['LED'].tolist()
            imu_led = {}
            imu_on = 1
            imu_off = 1
            for indx, item in enumerate(led_array):
                if item != 0 and led_array[indx - 1] == 0:
                    imu_led['{}LEDon'.format(imu_on)] = sample_array[indx]
                    imu_on += 1
                elif item == 0 and led_array[indx - 1] != 0:
                    if ledoff:
                        imu_led['{}LEDoff'.format(imu_off)] = sample_array[indx]
                        imu_off += 1
                    else:
                        continue

            # save data to sync_dict
            sync_dict['imu'] = imu_led

        else:
            print('IMU file not given or found.')

        print('Extraction complete! It took {:.2f} seconds.'.format(time.time() - t))

        # pack sync_dict into a df and save it to a .pkl file
        export_sync_df = pd.DataFrame(index=sync_dict['imec0'].keys(), columns=sync_dict.keys())
        for data_key in sync_dict.keys():
            for event_key in sync_dict['imec0'].keys():
                if event_key in sync_dict[data_key].keys():
                    export_sync_df.loc[event_key, data_key] = sync_dict[data_key][event_key]

        with open('{}.pkl'.format(self.sync_df), 'wb') as df:
            pickle.dump(export_sync_df, df)

        print('Sync file saved at {}.'.format(self.sync_df))
