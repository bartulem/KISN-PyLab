# -*- coding: utf-8 -*-

"""

@author: bartulem

Get spike times by splitting clusters.

"""

import pandas as pd
import numpy as np
import os
import sys
import scipy.io as sio
import pickle


class sst:
    
    # Initializer / Instance Attributes
    def __init__(self, thedir):
        self.thedir = thedir


    def splitClusters(self, **kwargs):
        
        '''
        Parameters
        ----------
        **kwargs: dictionary
        onesession : boolean (0/False or 1/True)
            If "False" - splits multiple sessions, otherwise - converts spike times of one session to seconds; dafault to "True".
        minspikes : int/float
            The minimum number of spikes; defaults to 100.
        pklfile : str
            Complete path to the .pkl file with total file lengths; defaults to 0.
        nchan : int/float
            Total number of channels on the NPX probe, for Probe3b should be 385; defaults to 385.
        ----------
        '''

        # test that all the prerequisites are there
        if(not os.path.exists(self.thedir)):
            print('Could not find directory {}, try again.'.format(self.thedir))
            sys.exit()

        # valid values for Booleans
        validBools = [0, False, 1, True]

        onesession = [kwargs['onesession'] if 'onesession' in kwargs.keys() and kwargs['onesession'] in validBools else 1][0]
        nchan = int([kwargs['nchan'] if 'nchan' in kwargs.keys() and type(kwargs['nchan']) == int or type(kwargs['nchan']) == float else 385][0])
        minspikes = [kwargs['minspikes'] if 'minspikes' in kwargs.keys() and type(kwargs['minspikes']) == int or type(kwargs['minspikes']) == float else 100][0]
        pklfile = [kwargs['pklfile'] if 'pklfile' in kwargs.keys() and type(kwargs['pklfile']) == str else 0][0]
    
        # load the appropriate spike data
        clusterInfo = pd.read_csv('{}\cluster_info.tsv'.format(self.thedir), sep="\t")  # info about all clusters
        spikeClusters = np.load('{}\spike_clusters.npy'.format(self.thedir))  # cluster IDs of all the spikes
        spikeTimes = np.load('{}\spike_times.npy'.format(self.thedir))  # spike times of all the clusters
    
        # check if one or more files require splitting
        if(not onesession):   

            # load the .pkl file with total file lengths
            file = open('{}\{}'.format(self.thedir, pklfile), 'rb')
            fileLengths = pickle.load(file)
            file.close()
        
            # creat a big dict where all the cells go
            allSpikeData = {}
    
            # get spikes from every good cluster and save them as spike times according to each session start
            for indx in range(clusterInfo.shape[0]):
                if(clusterInfo.loc[indx, 'group'] == 'good'):
                    clusterIndices = np.where(spikeClusters == clusterInfo.loc[indx, 'id'])[0]
                    spikesAllSessions = np.take(spikeTimes, clusterIndices)
         
                    # creat spikeDict and put each spike in the appropriate session
                    spikesDict = {'session_{}'.format(x+1): [] for x in range(len(fileLengths.keys())-1)}
                    for aspike in spikesAllSessions:
                        truth = True
                        while(truth):
                            for xx in range(len(fileLengths.keys())-1):
                                lowerBound = fileLengths['totalLEnChangepoints'][xx]//nchan
                                upperBound = fileLengths['totalLEnChangepoints'][xx+1]//nchan
                                if(lowerBound <= aspike < upperBound):
                                    spikesDict['session_{}'.format(xx+1)].append((aspike-lowerBound)/3e4)
                                    truth = False
                
                    print('The Phy cluster ID is {} (it has a total of {} spikes).'.format(clusterInfo.loc[indx, 'id'], clusterInfo.loc[indx, 'n_spikes']))
                    print('Splitting spikes in {} sessions produced the following results:'.format(len(fileLengths.keys())-1))
                    for asession in spikesDict.keys():
                        print('{} has {} spikes.'.format(asession, len(spikesDict[asession])))
                    print('In total, {} spikes have been accounted for.'.format(sum([len(spikesDict[key]) for key in spikesDict.keys()])))
                
                    allSpikeData['cell{}_ch{}'.format(clusterInfo.loc[indx, 'id'], clusterInfo.loc[indx, 'ch'])] = spikesDict
                
            # save spike .mat files (only if there's more than *minspikes* spk/session!)
            for session in range(len(fileLengths.keys())-1):
                path = r'{}\session{}'.format(self.thedir, session+1)
                if(not os.path.exists(path)):
                    os.makedirs(path)
            
                cellcount = 0
                for acell in allSpikeData.keys():
                    if(len(allSpikeData[acell]['session_{}'.format(session+1)]) > minspikes):
                        sio.savemat(path + '\\' + acell + '.mat', {'cellTS': np.array(allSpikeData[acell]['session_{}'.format(session+1)])}, oned_as='column')
                        cellcount += 1

                print('In session {}, there are {} good clusters (above {} spikes).'.format(session+1, cellcount, minspikes))
            
        else:
        
            # creat a big dict where all the cells go
            allSpikeData = {}

            # get spikes from every good cluster and save them as spike times according to session start
            for indx in range(clusterInfo.shape[0]):
                if(clusterInfo.loc[indx, 'group'] == 'good'):
                    clusterIndices = np.where(spikeClusters == clusterInfo.loc[indx, 'id'])[0]
                    spikesAll = np.take(spikeTimes, clusterIndices)
                
                    print('The Phy cluster ID is {} (it has a total of {} spikes).'.format(clusterInfo.loc[indx, 'id'], clusterInfo.loc[indx, 'n_spikes']))
                
                    spikes = []
                    for aspike in spikesAll:
                        spikes.append(aspike/3e4)
                    
                    allSpikeData['cell{}_ch{}'.format(clusterInfo.loc[indx, 'id'], clusterInfo.loc[indx, 'ch'])] = spikes

            # save spike .mat files (only if there's more than *minspikes* spk/session!)
            path = r'{}\session1'.format(self.thedir)
            if(not os.path.exists(path)):
                os.makedirs(path)
            
            cellcount = 0
            for acell in allSpikeData.keys():
                if(len(allSpikeData[acell]) > minspikes):
                    sio.savemat(path + '\\' + acell + '.mat', {'cellTS': np.array(allSpikeData[acell])}, oned_as='column')
                    cellcount += 1

            print('In this session, there are {} good clusters (above {} spikes).'.format(cellcount, minspikes))
        