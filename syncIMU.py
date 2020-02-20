# -*- coding: utf-8 -*-

"""

@author: bartulem

Check if IMU data syncs well with Npx data.

"""

import pandas as pd
from scipy.stats import linregress
import matplotlib.pyplot as plt
import numpy as np
import pickle


class imuS:

    # Initializer / Instance Attributes
    def __init__(self, txtsIMU, txtsNPX):
        self.txtsIMU = txtsIMU.replace('\\', '/')
        self.txtsNPX = txtsNPX.replace('\\', '/')

    def syncMilis(self):

        # load the files as df and add the imu header
        for imu, npx in zip(self.txtsIMU, self.txtsNPX):
            print('IMU file: {}, NPX file: {}'.format(imu, npx))
            npx_df = pd.read_csv('{}'.format(npx), sep='-', header=None, index_col=0)
            imu_df = pd.read_csv('{}'.format(imu), sep=',', header=None)
            imu_df.columns = ['loop.starttime (ms)', 'sample.time (ms)', 'acc.x', 'acc.y', 'acc.z', 'gyr.x', 'gyr.y', 'gyr.z', 'mag.x', 'mag.y', 'mag.z', 'LED', 'sys.cal', 'gyr.cal', 'acc.cal', 'mag.cal']

            # get LED indices
            samplearray = imu_df['sample.time (ms)'].tolist()
            ledarray = imu_df['LED'].tolist()
            ledON = []
            for indx, item in enumerate(ledarray):
                if (item != 0 and ledarray[indx - 1] == 0):
                    ledON.append(samplearray[indx])
            ledON.pop(-1)

            ledONtimesIMU = np.array(ledON)
            ledONtimesNPX = npx_df.iloc[1:-1, 0].to_numpy()

            print('The difference between NPX and IMU total session time is {:.3f} ms'.format((ledONtimesNPX[-1] - ledONtimesNPX[0] + 1) / 30 - (ledONtimesIMU[-1] - ledONtimesIMU[0] + 1)))

            # get the intercept and slope
            slope, intercept, r_value, p_value, std_err = linregress(ledONtimesIMU - ledONtimesIMU[0], (ledONtimesNPX - ledONtimesNPX[0]) / 30)
            print('The teensy sampling rate is {:.2f} Hz and the delay between NPX and IMU starts is {:.2f} ms'.format(slope * 1e3, intercept))

            # plot the result with regression line specifics
            fig, ax = plt.subplots(1, 1)
            ax.scatter(ledONtimesIMU - ledONtimesIMU[0], (ledONtimesNPX - ledONtimesNPX[0]) / 30, color='#000000')
            ax.plot(ledONtimesIMU - ledONtimesIMU[0], (ledONtimesIMU - ledONtimesIMU[0] * slope) + intercept, ls='-', color='#cc4f38', label='y={:.6f}x+{:.6f}'.format(slope, intercept))
            ax.legend(loc='best')
            ax.set_title('Regressing Tennsy on Npx')
            ax.set_xlabel('Npx (ms)')
            ax.set_ylabel('Teensy (ms)')
            plt.show()

            # save IMU data to file
            fileIMU = open('{}.pkl'.format(imu[:-4]), 'wb')
            pickle.dump(imu_df, fileIMU)
            fileIMU.close()
