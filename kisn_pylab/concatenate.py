# -*- coding: utf-8 -*-

"""

@author: bartulem

Concatenate NPX recordings (ap|lf) into one .bin file.

This script should be run if multiple sessions need to be merged (before running Kilosort2).
The primary reason for merging is the desire to keep the same cell IDs across the recorded
sessions. It should be said that there are other (less intuitive) ways to preserve cell IDs
and that merging is not always a good idea (e.g. if the sessions are separated by many hours).
I offer two options for conducting the merging: (1) through Python (less convenient) which
works well for smaller files (<10 Gb) as they need to be loaded into memory (I recommend
at least 64 Gb RAM), (2) through the CMD prompt/terminal (recommended) which works well for
larger files (>10 Gb) as it does not require a lot of memory. The end result of the process
is a large .bin file (concatenated recorded files) and a smaller binary .pkl file which stores
the lengths of individual sessions (in terms of samples) and their change-points in the newly
created concatenated file.

!NB: If you are concatenating e.g. three sessions, you may want to consider giving them the same name
with the 's1', 's2' and 's3' abbreviation as the distinctive factor because your input is the
directory where the files are, so you don't specify the order the files should be concatenated in.
However, if you use the same name with 's1', 's2' and 's3' being the difference, the code will
set the order right. The script prints the order of the to-be-concatenated files, so pay attention!

"""

import os
import sys
import numpy as np
from tqdm.notebook import tqdm
import time
import gc
import pickle


class Concat:

    # initializer / instance attributes
    def __init__(self, file_dir, new_file_name, pkl_len):
        self.file_dir = file_dir
        self.new_file_name = new_file_name
        self.pkl_len = pkl_len

    def concat_npx(self, **kwargs):

        """
        Inputs
        ----------
        **kwargs: dictionary
        cmd_prompt : boolean (0/False or 1/True)
            Run the merging through Python (0) or the command prompt (1); defaults to 1.
        nchan : int/float
            Total number of channels on the NPX probe, for probe 3B2 should be 385; defaults to 385.
        npx_sampling_rate : int/float
            The sampling rate of the NPX system; defaults to 3e4.
        file_type : str
            For spikes, choose 'ap', for LFP choose 'lf'; defaults to 'ap'.
        ----------

        Outputs
        ----------
        new_file_name : binary array
            The newly created concatenated data array; saved as .bin file.
        pkl_len : dictionary
            Information about change-points of the concatenated sessions; saved as .pkl file.
        ----------
        """

        # check that the directory is there
        if not os.path.exists(self.file_dir):
            print('Could not find directory {}, try again.'.format(self.file_dir))
            sys.exit()

        # valid values for booleans
        valid_booleans = [0, False, 1, True]

        cmd_prompt = kwargs['cmd_prompt'] if 'cmd_prompt' in kwargs.keys() and kwargs['cmd_prompt'] in valid_booleans else 1
        nchan = int(kwargs['nchan'] if 'nchan' in kwargs.keys() and (type(kwargs['nchan']) == int or type(kwargs['nchan']) == float) else 385)
        npx_sampling_rate = float(kwargs['npx_sampling_rate'] if 'npx_sampling_rate' in kwargs.keys() else 3e4)
        file_type = kwargs['file_type'] if 'file_type' in kwargs.keys() and (kwargs['file_type'] == 'ap' or kwargs['file_type'] == 'lf') else 'ap'

        # print to see if order of concatenation is right
        file_lst = []
        print('The files will be concatenated in the following order:')

        for afile in os.listdir(self.file_dir):
            if file_type in afile and 'bin' in afile:
                print(afile)
                file_lst.append(afile)

        if len(file_lst) == 0:
            print('No appropriate files in this directory!')
            sys.exit()

        # get files together with paths and read them
        file_paths = []
        file_lengths = {'total_len_changepoints': [0]}

        for afile in file_lst:

            # create absolute paths for every file
            npx_file = '{}{}{}'.format(self.file_dir, os.sep, afile)
            file_paths.append(npx_file)

            # get raw 1D array length for each recording
            npx_recording = np.memmap(npx_file, mode='r', dtype=np.int16, order='C')
            file_lengths[npx_file] = npx_recording.shape[0]

            if len(file_lengths['total_len_changepoints']) == 1:
                file_lengths['total_len_changepoints'].append(npx_recording.shape[0])
            else:
                file_lengths['total_len_changepoints'].append(npx_recording.shape[0] + file_lengths['total_len_changepoints'][-1])

            print('Found file: {} with total length {}, '
                  'or {} samples, or {} minutes.'.format(npx_file, npx_recording.shape[0],
                                                         npx_recording.shape[0] // nchan,
                                                         round(npx_recording.shape[0] // nchan / (npx_sampling_rate*60), 2)))

            # delete the map object from memory
            del npx_recording
            gc.collect()

        # save the file_lengths dictionary
        with open('{}'.format(self.pkl_len), 'wb') as save_dict:
            pickle.dump(file_lengths, save_dict)

        # concatenate
        print('Concatenating files, please be patient - this could take >1 hour.')

        start_time = time.time()

        if not cmd_prompt:

            # create new empty binary file & memory map array to load data into
            with open(self.new_file_name, 'wb'):
                pass

            new_array_total_len = sum([file_lengths[akey] for akey in file_lengths.keys() if akey != 'total_len_changepoints'])
            concatenated_file = np.memmap(self.new_file_name, dtype=np.int16, mode='r+', shape=(new_array_total_len,), order='C')

            print('The concatenated file has total length of {}, '
                  'or {} samples, or {} minutes.'.format(concatenated_file.shape[0], concatenated_file.shape[0] // nchan,
                                                         round(concatenated_file.shape[0] // nchan / (npx_sampling_rate*60), 2)))

            # give it a 1s break
            time.sleep(1)

            # fill it with data
            counter = 0
            for onefile in tqdm(file_paths):
                npx_recording = np.memmap(onefile, mode='r', dtype=np.int16, order='C')
                if counter == 0:
                    concatenated_file[:file_lengths[onefile]] = npx_recording
                    counter += file_lengths[onefile]
                else:
                    concatenated_file[counter:counter + file_lengths[onefile]] = npx_recording
                    counter += file_lengths[onefile]

                # delete the map object from memory
                del npx_recording
                gc.collect()

            # delete the concatenated map object from memory
            del concatenated_file
            gc.collect()

        else:

            # give it a 1s break
            time.sleep(1)

            # get all files in a command
            command = 'copy /b '
            for afileindx, afile in enumerate(file_paths):
                if afileindx < (len(file_paths) - 1):
                    command += '{} + '.format(afile)
                else:
                    command += '{} '.format(afile)
            command += '"{}"'.format(self.new_file_name)

            # outsource command
            os.system('cmd /c "{}"'.format(command))

        print('Concatenation complete! It took {:.2f} minutes.\n'.format((time.time() - start_time) / 60))
