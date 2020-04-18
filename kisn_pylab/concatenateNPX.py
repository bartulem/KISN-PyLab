# -*- coding: utf-8 -*-

"""

@author: bartulem

Concatenate NPX sessions into one file for running Kilosort2.

This script should be run if multiple sessions need to be merged before running Kilosort2.
The primary reason for merging is the desire to keep the same cell IDs across the recorded
sessions. It should be said that there are other (less intuitive) ways to preserve cell IDs
and that merging is not always a good idea (e.g. if the sessions are separated by many hours).
I offer two options for conducting the merging: (1) thorough Python (less convenient) which
works well for smaller files (<10 Gb) as they need to be loaded into memory (I recommend
at least 64 Gb RAM), (2) through the CMD prompt/terminal (recommended) which works well for
larger files (>10 Gb) as it does not require a lot of memory. The end result of the process
is a large .bin file (concatenated recorded files) and a smaller binary .pkl file which stores
the lengths of individual sessions (in terms of samples) and their changepoints in the newly
created concatenated file.

"""

import os
import sys
import numpy as np
from tqdm import tqdm
import time
import gc
import pickle


class Concat:

    # initializer / instance attributes
    def __init__(self, file_dir, new_file_name, future_pkl):
        self.file_dir = file_dir
        self.new_file_name = new_file_name
        self.future_pkl = future_pkl

    def concat_npx(self, **kwargs):

        """
        Parameters
        ----------
        **kwargs: dictionary
        cmd_prompt : boolean (0/False or 1/True)
            Run the merging through Python (0) or the command prompt (1); defaults to 1.
        nchan : int/float
            Total number of channels on the NPX probe, for Probe3b should be 385; defaults to 385.
        ----------
        """

        # test that all the prerequisites are there
        if not os.path.exists(self.file_dir):
            print('Could not find directory {}, try again.'.format(self.file_dir))
            sys.exit()

        # valid values for booleans
        valid_bools = [0, False, 1, True]

        cmd_prompt = [kwargs['cmd_prompt'] if 'cmd_prompt' in kwargs.keys() and kwargs['cmd_prompt'] in valid_bools else 1][0]
        nchan = int([kwargs['nchan'] if 'nchan' in kwargs.keys() and type(kwargs['nchan']) == int or type(kwargs['nchan']) == float else 385][0])

        # get files together with paths and read them
        file_paths = []
        file_lengths = {'total_len_changepoints': [0]}
        for afile in os.listdir(self.file_dir):
            if 'ap' in afile and 'bin' in afile:
                npx_file = '{}{}{}'.format(self.file_dir, os.sep, afile)
                file_paths.append(npx_file)
                npx_recording = np.memmap(npx_file, mode='r', dtype=np.int16, order='C')
                file_lengths[npx_file] = npx_recording.shape[0]
                if len(file_lengths['total_len_changepoints']) == 1:
                    file_lengths['total_len_changepoints'].append(npx_recording.shape[0])
                else:
                    file_lengths['total_len_changepoints'].append(npx_recording.shape[0] + file_lengths['total_len_changepoints'][-1])

                print('Found file: {} with total length {}, or {} samples, or {} minutes.'.format(npx_file, npx_recording.shape[0], npx_recording.shape[0] // nchan, round(npx_recording.shape[0] // nchan / 18e5, 2)))

                # delete the big memmap obj from memory
                del npx_recording
                gc.collect()
            else:
                print('No files in this directory!')
                sys.exit()

        if not cmd_prompt:

            t = time.time()

            # create new empty bin file & memmap array to load data into
            with open(self.new_file_name, 'wb'):
                pass

            new_array_total_len = sum([file_lengths[akey] for akey in file_lengths.keys() if akey != 'total_len_changepoints'])
            conc_file = np.memmap(self.new_file_name, dtype=np.int16, mode='r+', shape=(new_array_total_len,), order='C')
            print('The concatenated file has total length of {}, or {} samples, or {} minutes.'.format(conc_file.shape[0], conc_file.shape[0] // nchan, round(conc_file.shape[0] // nchan / 18e5, 2)))

            # fill it with data
            print('Concatenating files, please be patient - this can take up to several minutes.')
            time.sleep(1)

            counter = 0
            for onefile in tqdm(file_paths):
                npx_recording = np.memmap(onefile, mode='r', dtype=np.int16, order='C')
                if counter == 0:
                    conc_file[:file_lengths[onefile]] = npx_recording
                    counter += file_lengths[onefile]
                else:
                    conc_file[counter:counter + file_lengths[onefile]] = npx_recording
                    counter += file_lengths[onefile]

                # delete the big memmap obj from memory
                del npx_recording
                gc.collect()

            # delete the big memmap obj from memory
            del conc_file
            gc.collect()

            print('Concatenation complete! It took {:.2f} seconds.'.format(time.time() - t))

        else:

            print('Concatenating files, please be patient - this can take up to several minutes.')

            # get all files in a command
            command = 'copy /b '
            for afileindx, afile in enumerate(file_paths):
                if afileindx < (len(file_paths) - 1):
                    command += '{} + '.format(afile)
                else:
                    command += '{} '.format(afile)
            command += '"{}"'.format(self.new_file_name)

            # outsource command and keep time
            t = time.time()
            os.system('cmd /c "{}"'.format(command))
            print('Concatenation complete! It took {:.2f} seconds.'.format(time.time() - t))

        # save the file_lengths dictionary
        save_dict = open('{}'.format(self.future_pkl), 'wb')
        pickle.dump(file_lengths, save_dict)
        save_dict.close()
