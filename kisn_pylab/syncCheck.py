# -*- coding: utf-8 -*-

"""

@author: bartulem

Estimate how well different data streams (tracking/IMU) are synced with neural recordings.

This script utilizes the LED timestamp occurrences across different data streams and estimates how well they are linearly related.
Specifically, the tracking/IMU data is split (train/test) and used to predict the imec LED occurrences in the test set. The main
results of this analysis are printed (but they can also be plotted). Importantly, the code also provides an estimate for the true
capture rate of the tracking cameras (instead of the theoretical one). In order to make the figures in this plot, you need to
install *plotly* which enables interactive plotting (and we may want to see, for any given test point, what the quality of the
prediction was).

"""

import os
import numpy as np
import pickle
from inhouseLRegression import LinRegression
from plotly.subplots import make_subplots
import plotly.graph_objects as go


class Sync:

    # initializer / instance attributes
    def __init__(self, pkl_files):
        self.pkl_files = pkl_files

    def estimate_sync_quality(self, **kwargs):

        """
        Parameters
        ----------
        **kwargs: dictionary
        npx_sampling_rate : int/float
            The sampling rate of the NPX system; defaults to 3e4.
        to_plot : boolean (0/False or 1/True)
            To plot or not to plot y_test and y_test_prediction scatter plot; defaults to 0.
        ----------
        """
        # valid values for booleans
        valid_bools = [0, False, 1, True]

        to_plot = [kwargs['to_plot'] if 'to_plot' in kwargs.keys() and kwargs['to_plot'] in valid_bools else 0][0]
        npx_sampling_rate = float([kwargs['npx_sampling_rate'] if 'npx_sampling_rate' in kwargs.keys() else 3e4][0])
    
        # save all the data in a dictionary where file names are keys
        sync_data = {}
        for pkl_file in self.pkl_files:
            if os.path.exists(pkl_file):
                with open(pkl_file, 'rb') as one_file:
                    sync_data[pkl_file] = pickle.load(one_file)
            else:
                print('File: {} does not exist!'.format(pkl_file))

        # go through each file and estimate sync quality
        for file_key in sync_data.keys():

            print('Working on file: {}.'.format(file_key))

            temp_df = sync_data[file_key]

            # separate imu/tracking from probe data
            imec_data_cols = []
            other_data_cols = []
            for data_stream in temp_df.columns:
                if 'imec' in data_stream:
                    imec_data_cols.append(data_stream)
                else:
                    other_data_cols.append(data_stream)

            for data_stream in other_data_cols:
                data_stream_dict = {}
                for imec_data in imec_data_cols:

                    # reduce the dataframe to only include LED occurrences
                    imec_data_col = temp_df.columns.tolist().index(imec_data)
                    data_stream_col = temp_df.columns.tolist().index(data_stream)
                    reduced_df = temp_df.iloc[2:-2, [imec_data_col, data_stream_col]]

                    # transform data for imec (s or ms) depending on data stream (tracking/imu) and add extra_data to be predicted if necessary
                    if data_stream == 'tracking':
                        # convert imec sample numbers into seconds
                        reduced_df.iloc[:, 0] = reduced_df.iloc[:, 0]/npx_sampling_rate
                        extra_data = temp_df.loc['TTL input start', 'tracking']
                    else:
                        # convert imec sample numbers into milliseconds (subtracted from the first LED occurrence to zero it & do the same for IMU data)
                        reduced_df.iloc[:, 0] = (reduced_df.iloc[:, 0] - reduced_df.iloc[0, 0])/(npx_sampling_rate/1e3)
                        reduced_df.iloc[:, 1] = reduced_df.iloc[:, 1] - reduced_df.iloc[0, 1]
                        extra_data = 0

                    # regress
                    regress_class = LinRegression(reduced_df)
                    data_stream_dict[imec_data] = regress_class.split_train_test_and_regress(xy_order=[1, 0], extra_data=extra_data)

                # print relevant information and save empirical frame rate in the last column of .pkl file
                differences_dict = {}
                if data_stream == 'tracking':
                    empirical_frame_rates = []
                    for imec_data in imec_data_cols:
                        true_predicted_differences = data_stream_dict[imec_data]['y_test'] - data_stream_dict[imec_data]['y_test_predictions']
                        differences_dict[imec_data] = true_predicted_differences
                        print('According to imec probe {}:'.format(int(imec_data[-1])))
                        print('The offset between the real start of tracking and the TTL start tracking pulse is {:.2f} ms.'.format(data_stream_dict[imec_data]['extra_data_predictions'][0]*1e3))
                        print('The differences between y_test and y_test_predictions are: median - {:.2f} ms, mean - {:.2f} ms, max - {:.2f} ms.'.format(np.nanmedian(np.abs(true_predicted_differences))*1e3, np.nanmean(np.abs(true_predicted_differences))*1e3, np.nanmax(np.abs(true_predicted_differences))*1e3))
                        emp_fr_rate = round(1/data_stream_dict[imec_data]['slope'], 5)
                        empirical_frame_rates.append(emp_fr_rate)
                        print('The estimated empirical frame rate is {}.'.format(emp_fr_rate))

                    # re-save .pkl file with new frame rate
                    temp_df['Est true fps'] = np.zeros(temp_df.shape[0])
                    temp_df.iloc[0, -1] = np.nanmean(empirical_frame_rates)
                    print('Saving empirical capture rate: {} fps.'.format(np.nanmean(empirical_frame_rates)))

                    with open('{}'.format(file_key), 'wb') as df:
                        pickle.dump(temp_df, df)

                else:
                    for imec_data in imec_data_cols:
                        true_predicted_differences = data_stream_dict[imec_data]['y_test'] - data_stream_dict[imec_data]['y_test_predictions']
                        differences_dict[imec_data] = true_predicted_differences
                        print('According to imec probe {}:'.format(int(imec_data[-1])))
                        print('The difference between NPX and IMU total session time is {:.3f} ms.'.format(reduced_df.iloc[-1, 0] - reduced_df.iloc[-1, 1]))
                        print('Teensy sampling rate is {:.2f} Hz and the delay between NPX and IMU starts is {:.2f} ms.'.format(data_stream_dict[imec_data]['slope']*1e3, data_stream_dict[imec_data]['intercept']))
                        print('The differences between y_test and y_test_predictions are: median - {:.2f} ms, mean - {:.2f} ms, max - {:.2f} ms.'.format(np.nanmedian(np.abs(true_predicted_differences)), np.nanmean(np.abs(true_predicted_differences)), np.nanmax(np.abs(true_predicted_differences))))

                # plot y_test and y_test_prediction differences
                if to_plot:
                    colors = {'imec0': '#EE5C42', 'imec1': '#1E90FF'}
                    fig = make_subplots(rows=len(data_stream_dict.keys()), cols=2)
                    fig.update_layout(height=1000, width=1000, plot_bgcolor='#FFFFFF', title='{}'.format(data_stream), showlegend=True)
                    fig_order = [[1, 2], [3, 4]]
                    for indx, imec_data in enumerate(imec_data_cols):
                        fig.append_trace(go.Scatter(x=data_stream_dict[imec_data]['y_test'], y=data_stream_dict[imec_data]['y_test_predictions'], mode='markers', name=imec_data, marker=dict(color=colors[imec_data], size=5)), row=indx+1, col=1)
                        fig['layout']['xaxis{}'.format(fig_order[indx][0])].update(title='y_test')
                        fig['layout']['yaxis{}'.format(fig_order[indx][0])].update(title='y_test_predictions')

                        fig.append_trace(go.Histogram(x=differences_dict[imec_data], name='{} errors'.format(imec_data), marker_color=colors[imec_data], opacity=.75), row=indx+1, col=2)
                        fig['layout']['xaxis{}'.format(fig_order[indx][1])].update(title='true - predicted')
                        fig['layout']['yaxis{}'.format(fig_order[indx][1])].update(title='count')
                    fig.show()
