# -*- coding: utf-8 -*-

"""

@author: bartulem

Do the regression to calculate 'Npx pred (sec)'. 

"""

from scipy.stats import linregress
import matplotlib.pyplot as plt


class lr:

    # Initializer / Instance Attributes
    def __init__(self, syncData):
        self.syncData = syncData
        

    def linreg(self):
        for txtdir in self.syncData:
            for afile in self.syncData[txtdir].keys():
                print('Working on file {} in directory {}'.format(afile, txtdir))

                # get the intercept and slope
                slope, intercept, r_value, p_value, std_err = linregress(self.syncData[txtdir][afile].iloc[1:-1, 3], self.syncData[txtdir][afile].iloc[1:-1, 1])
            
                # plot the result with regression line specifics
                fig, ax = plt.subplots(1, 1)
                ax.scatter(self.syncData[txtdir][afile].iloc[1:-1, 3], self.syncData[txtdir][afile].iloc[1:-1, 1], color='#000000')
                ax.plot(self.syncData[txtdir][afile].iloc[1:-1, 3], (self.syncData[txtdir][afile].iloc[1:-1, 3]*slope)+intercept, ls='-', color='#cc4f38', label='y={:.6f}x+{:.6f}'.format(slope, intercept))
                ax.legend(loc='best')
                ax.set_title('Regressing Npx on Opti')
                ax.set_xlabel('Opti (true frame)')
                ax.set_ylabel('Npx true (sec)')
                plt.show()
            
                # now go back to the df, and calculate the last 2 columns
                self.syncData[txtdir][afile].loc[:, 'Npx pred (sec)'] = (self.syncData[txtdir][afile].loc[:, 'Opti (true frame)']*slope)+intercept
                self.syncData[txtdir][afile].loc[:, 'Diff true-pred Npx (ms)'] = (self.syncData[txtdir][afile].loc[:, 'Npx true (sec)'] - self.syncData[txtdir][afile].loc[:, 'Npx pred (sec)'])*1e3  # multiply with 1e3 to convert to ms
            
                # calculate maxError and offset and print them
                maxError = self.syncData[txtdir][afile].iloc[1:-1, -2].abs().max()
                offset = self.syncData[txtdir][afile].iloc[0, -2]           
                print('The offset between the real start of tracking and the TTL start tracking pulse is {:.2f} ms, the max error in predicting LED appearances from timestamps is {:.2f} ms and the estimated true camera frame rate is {:.5f} fps!'.format(offset, maxError, 1/slope))
            
                # store the est. true frame rate in the first row of the last df column
                self.syncData[txtdir][afile].iloc[0, -1] = round(1/slope, 5)
            
                # save df to csv file for posterity
                self.syncData[txtdir][afile].to_csv(r'{}\{}.csv'.format(txtdir, afile), sep=';', header=True, index=True)
            
