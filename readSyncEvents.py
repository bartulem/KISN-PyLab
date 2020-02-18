# -*- coding: utf-8 -*-

"""

@author: bartulem

Neuropixel recordings are saved into a *1D binary vector*, from a 2D array organised as 385 rows (channels) x n columns (samples). 
The data are written to file from this matrix in column-major (F) order, i.e., the first sample in the recording was written to file 
for every channel, then the second sample was written for every channel, etc.

Neuropixel raw data is provided as an int16 binary. Neuropixel ADCs are 10 bits, with a range of -0.6 to 0.6 V, 
and acquisition was at 500x gain, yielding a resolution of 2.34 µV/bit.
To obtain readings in µV, you should multiply the int16 values by 2.34.

"""  

import numpy as np
from tqdm import tqdm
import gc
import time


class read:
   
    # Initializer / Instance Attributes
    def __init__(self, npxFile, txtFile):
        self.npxFile = npxFile
        self.txtFile = txtFile


    def readSE(self, **kwargs):
        
        '''
        Parameters
        ----------
        **kwargs: dictionary
        nchan : int/float
            Total number of channels on the NPX probe, for Probe3b should be 385; defaults to 385.
        ----------
        '''
        
        nchan = int([kwargs['nchan'] if 'nchan' in kwargs.keys() and type(kwargs['nchan']) == int or type(kwargs['nchan']) == float else 385][0])

        print('The file to be worked on is:', self.npxFile)
        print('Extracting sync data from file, please be patient - this can take up to several minutes.')
        time.sleep(2)
        
        # memmaps are used for accessing small segments of large files on disk, without reading the entire file into memory.
        npxRecording = np.memmap(self.npxFile, mode='r', dtype=np.int16, order='C')
        
        # integer divide the length of the recording by channel num. to get number of samples
        npxSamples = len(npxRecording)//nchan
        
        # reshape the array such that channels are rows and samples are columns
        npxRecording = npxRecording.reshape((nchan, npxSamples), order='F')
        
        # get the sync data in a separate array
        syncData = npxRecording[nchan-1, :]
        
        # export sync events to .txt file
        changepoints = []
        for inxSync, itemSync in tqdm(enumerate(syncData)):
            if(len(changepoints) == 0):
                if(itemSync != 0 and syncData[inxSync-1] == 0):
                    changepoints.append(inxSync)
            else:
                if(itemSync != 0 and syncData[inxSync-1] == 0):
                    changepoints.append(inxSync-1)
                elif(itemSync == 0 and syncData[inxSync-1] != 0):
                    changepoints.append(inxSync)
        changepoints[-1] = changepoints[-1]-1            


        counterON = 1
        counterOFF = 1
        with open(self.txtFile, 'w') as txtfile:
            txtfile.write('{:15} - {} \n'.format('Session start', 0))
            for indx, timestamp in enumerate(changepoints):
                if(indx == 0):
                    txtfile.write('{:15} - {} \n'.format('TTL input start', timestamp))
                elif(indx != 0 and indx != len(changepoints)-1 and indx % 2 != 0):
                    txtfile.write('{:15} - {} \n'.format('{}LEDon'.format(counterON), timestamp))
                    counterON += 1
                elif(indx != 0 and indx != len(changepoints)-1 and indx % 2 == 0):
                    # txtfile.write('{:15} - {} \n'.format('{}LEDoff'.format(counterOFF), timestamp))
                    counterOFF += 1
                else:
                    txtfile.write('{:15} - {} \n'.format('TTL input stop', timestamp))
            txtfile.write('{:15} - {}'.format('Session stop', len(syncData)))
        
        # delete the memmap obj from memory
        del npxRecording
        gc.collect()     
        print('Extraction complete!')
