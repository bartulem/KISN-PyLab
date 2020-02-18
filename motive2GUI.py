# -*- coding: utf-8 -*-

"""

@author: bartulem

Convert the labeled tracking .csv to .pkl for loading into the GUI.

"""

import pandas as pd
import os
import sys
import csv
import time
import pickle


class mtg:
    
    # Initializer / Instance Attributes
    def __init__(self, thecsv, thetxt):
        self.thecsv = thecsv
        self.thetxt = thetxt
        
        
    def csvTOpkl(self, **kwargs):
        
        '''
        Parameters
        ----------
        **kwargs: dictionary
        framerate : str (file name)
            The empirical camera frame rate for that session; defaults to 120 if no file is provided.
        samplingrate : int/float
            The sampling rate of the NPX system; defaults to 3e4.
        ----------
        '''
        
        # test that all the prerequisites are there
        if(not os.path.exists(self.thecsv)):
            print('Could not find {}, try again.'.format(self.thecsv))
            sys.exit()
            
        if(not os.path.exists(self.thetxt)):
            print('Could not find {}, try again.'.format(self.thetxt))
            sys.exit()
            
        print('Working on file: {}'.format(self.thecsv))
        t = time.time()
        
        framerate = float([pd.read_csv(kwargs['framerate'], sep=';', header=0, index_col=0).iloc[0, -1] if 'framerate' in kwargs.keys() else 120.][0])
        samplingrate = int([kwargs['samplingrate'] if 'samplingrate' in kwargs.keys() else 3e4][0])
        
        # get tracking data from .csv
        data = []
        with open(self.thecsv, 'r') as csvfile:
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
        timeStamps = {'startratcamtimestamp': 0, 'stopratcamtimestamp': 0, 'startsessiontimestamp': 0, 'stopsessiontimestamp': 0, 'starttrackingtimestamp': 0, 'stoptrackingtimestamp': 0, }
        timeStamps_df = pd.read_csv('{}'.format(self.thetxt), sep='-', header=None, index_col=0)
        timeStamps['starttrackingtimestamp'] = int(timeStamps_df.iloc[2, 0])
        timeStamps['stoptrackingtimestamp'] = int(timeStamps_df.iloc[-3, 0])
        timeStamps['stopsessiontimestamp'] = int(timeStamps_df.iloc[-1, 0])
        
        # load .csv file and get labelsdata
        labelsdict = {'Marker1': 0, 'Marker2': 1, 'Marker3': 2, 'Marker4': 3, 'Neck': 4, 'Back': 5, 'Ass': 6}
        labelsdata = pd.read_csv(self.thecsv, sep=',', nrows=2)
        labelsraw = labelsdata.iloc[1, :].tolist()[1:]
        labels = []
        for albaelind, alabel in enumerate(labelsraw):
            if(albaelind % 3 == 0):
                labels.append(labelsdict[alabel.split(':')[-1]])
        
        # get the final output dict ready (7 is the number of points, 5 is X, Y, Z, label, nans)
        final = {'points': [[[] for i in range(5)] for j in range(7)],
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
                                     'ass']},
                 'boundingboxscaleX': 0,
                 'boundingboxscaleY': 0,
                 'starttrackingtimestamp': timeStamps['starttrackingtimestamp'] / samplingrate,
                 'stoptrackingtimestamp': timeStamps['stoptrackingtimestamp'] / samplingrate,
                 'startsessiontimestamp': timeStamps['startsessiontimestamp'] / samplingrate,
                 'stopsessiontimestamp': timeStamps['stopsessiontimestamp'] / samplingrate,
                 'startratcamtimestamp': timeStamps['startratcamtimestamp'] / samplingrate,
                 'stopratcamtimestamp': timeStamps['stopratcamtimestamp'] / samplingrate,
                 'boundingboxrotation': 0,
                 'headXarray': [],
                 'framerate': framerate,
                 'headZarray': [],
                 'headYarray': [],
                 'headoriginarray': [],
                 'boundingboxtransX': 0,
                 'boundingboxtransY': 0}
        
        # the rest is Srikanth's
        CoOrdinatesOrder = [0, 2, 1]
        for i in range(7):
            tt = 0
            for j in CoOrdinatesOrder:
                X = [x[2 + 3 * i + j] for x in data]
                final['points'][i][tt] = ([(float(x) if x else -1000000) for x in X])
                tt = tt + 1
            final['points'][i][3] = [labels[i]] * frames
            final['points'][i][4] = [0.0] * frames

        for i in range(7):
            theList = final['points'][i][1]
            temp = [ind for ind in range(len(theList)) if theList[ind] == -1000000]
            for j in temp:
                final['points'][i][3][j] = -1000000
                final['points'][i][4][j] = -1000000
                
        # save result to file
        with open('{}.pkl'.format(self.thecsv[:-4]), 'wb') as f:
            # protocol 2 because the GUI doesn't regonize anything above
            pickle.dump(final, f, protocol=2)

        print('Finished, it took {:.2f} sec.'.format(time.time() - t))
