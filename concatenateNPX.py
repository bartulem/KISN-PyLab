# -*- coding: utf-8 -*-

"""

@author: bartulem

Script to concatenate Npx sessions into one file for running kilosort.

"""

import os
import sys
import numpy as np
from tqdm import tqdm
import time
import gc
import pickle


class concat:
   
    # Initializer / Instance Attributes
    def __init__(self, fileDir, newFileName):
        self.fileDir = fileDir
        self.newFileName = newFileName

   
    def concatNPX(self, **kwargs):
   
            '''
            Parameters
            ----------
            **kwargs: dictionary
            cmdPrompt : boolean (0/False or 1/True)
                Run the merging through Python or the command prompt; defaults to 1.
            nchan : int/float
                Total number of channels on the NPX probe, for Probe3b should be 385; defaults to 385.
            ----------
            '''
            
            # test that all the prerequisites are there
            if(not os.path.exists(self.fileDir)):
                print('Could not find directory {}, try again.'.format(self.fileDir))
                sys.exit()

            # valid values for Booleans
            validBools = [0, False, 1, True]

            cmdPrompt = [kwargs['cmdPrompt'] if 'cmdPrompt' in kwargs.keys() and kwargs['cmdPrompt'] in validBools else 1][0]
            nchan = int([kwargs['nchan'] if 'nchan' in kwargs.keys() and type(kwargs['nchan']) == int or type(kwargs['nchan']) == float else 385][0])
            
            # get files together with paths and read them 
            filePaths = []
            fileLengths = {'totalLEnChangepoints': [0]}
            for afile in os.listdir(self.fileDir):
                if('ap' in afile and 'bin' in afile):
                    npxFile = '{}\{}'.format(self.fileDir, afile)
                    filePaths.append(npxFile)
                    npxRecording = np.memmap(npxFile, mode='r', dtype=np.int16, order='C')
                    fileLengths[npxFile] = npxRecording.shape[0]
                    if(len(fileLengths['totalLEnChangepoints']) == 1):
                        fileLengths['totalLEnChangepoints'].append(npxRecording.shape[0])
                    else:
                        fileLengths['totalLEnChangepoints'].append(npxRecording.shape[0] + fileLengths['totalLEnChangepoints'][-1])
                        
                    print('Found file: {} with total length {}, or {} samples, or {} mins.'.format(npxFile, npxRecording.shape[0], npxRecording.shape[0]//nchan, round(npxRecording.shape[0]//nchan/18e5, 2)))
                    del npxRecording
                    gc.collect()
                else:
                    print('No files in this directory!')
                    sys.exit()
    
            if(not cmdPrompt):

                # create new empty bin file & memmap array to load data into
                with open(self.newFileName, 'wb'): 
                    pass
                
                newarrayTotalLen = sum([fileLengths[akey] for akey in fileLengths.keys() if akey != 'totalLEnChangepoints'])
                concFile = np.memmap(self.newFileName, dtype=np.int16, mode='r+', shape=(newarrayTotalLen,), order='C')
                print('The concatenated file has total length of {}, or {} samples, or {} mins.'.format(concFile.shape[0], concFile.shape[0]//nchan, round(concFile.shape[0]//nchan/18e5, 2)))
    
                # fill it with data
                print('Concatenating files, please be patient - this can take up to several minutes.')
                time.sleep(1)
    
                counter = 0
                for onefile in tqdm(filePaths):
                    npxRecording = np.memmap(onefile, mode='r', dtype=np.int16, order='C')
                    if(counter == 0):
                        concFile[:fileLengths[onefile]] = npxRecording
                        counter += fileLengths[onefile]
                    else:
                        concFile[counter:counter+fileLengths[onefile]] = npxRecording
                        counter += fileLengths[onefile]
                    del npxRecording
                    gc.collect()
            
                # delete the big memmap obj from memory
                del concFile
                gc.collect() 
    
                print('Concatenation complete!')
        
            else:
        
                # get all files in a command
                command = 'copy /b '
                for afileindx, afile in enumerate(filePaths):
                    if(afileindx < len(filePaths)-1):
                        command += '{} + '.format(afile)
                    else:
                        command += '{} '.format(afile)
                command += '"{}"'.format(self.newFileName)
        
                # outsource command and keep time
                t = time.time()
                os.system('cmd /c "{}"'.format(command))
                print('Concatenation took {:.2f} seconds.\n'.format(time.time() - t))
        
            # save the fileLength dict
            saveDict = open('{}.pkl'.format(self.newFileName[:-4]), 'wb')
            pickle.dump(fileLengths, saveDict)
            saveDict.close()