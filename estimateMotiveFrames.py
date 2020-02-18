# -*- coding: utf-8 -*-

"""

@author: bartulm

Estimate where in the Motive .tak files the LEDon events would be, assuming a 120fps acquisition rate.

"""

import os
import pandas as pd
import numpy as np


class emf:

    # Initializer / Instance Attributes
    def __init__(self, txtdirs):
        self.txtdirs = txtdirs
        

    def estimateMF(self):
    
        # save all the data in a dict where file names are keys
        syncData = {}
        for txtdir in self.txtdirs:
            syncData[txtdir] = {}
            for file in os.listdir(txtdir):
                if('.txt' in file):
                    syncData[txtdir][file[:-4]] = 0
    
        # open each file and get TTL input start, LEDon(s) and TTL input stop
        for txtdir in self.txtdirs:
            for afile in syncData[txtdir].keys():
                temp_df = pd.read_csv('{}\{}.txt'.format(txtdir, afile), sep='-', header=None, index_col=0)

                # remove start and end of recording
                temp_df = temp_df.drop(['{:16}'.format('Session start'), '{:16}'.format('Session stop')])
        
                # add extra columns to the df and give them appropriate names; also fill in second (Npx true (sec)) and third (Opti (est frame)) column
                for num in range(2, 8, 1):
                    if(num != 2 and num != 3):
                        temp_df[num] = np.zeros(temp_df.shape[0])
                    else:
                        if(num == 2):
                            temp_df[num] = temp_df.iloc[:, 0]/3e4
                        else:
                            temp_df[num] = (temp_df.iloc[:, 1]-temp_df.iloc[0, 1])*120.

                temp_df.columns = ['Npx true (sample)', 'Npx true (sec)', 'Opti (est frame)', 'Opti (true frame)', 'Npx pred (sec)', 'Diff true-pred Npx (ms)', 'Est true fps']
            
                print('For {} in {}, the estimated LEDon frames in the Motive .tak file are: {}'.format(afile, txtdir, temp_df['Opti (est frame)'].tolist()))
            
                syncData[txtdir][afile] = temp_df
            
        return syncData
                  
