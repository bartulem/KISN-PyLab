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
import sys
import numpy as np
import pickle
from kisn_pylab import regress
from plotly.subplots import make_subplots
import plotly.graph_objects as go


class Sync:

    # initializer / instance attributes
    def __init__(self, sync_pkls):
        self.sync_pkls = sync_pkls

    def estimate_sync_quality(self, **kwargs):

        """
        Parameters
        ----------
        **kwargs: dictionary
        npx_sampling_rate : int/float
            The sampling rate of the NPX system; defaults to 3e4.
        to_plot : boolean (0/False or 1/True)
            To plot or not to plot y_test and y_test_prediction scatter plot; defaults to 0.
        ground_probe : int
            In a multi probe setting, the probe other probes are synced to; defaults to 0.
        imu_files : list
            The list of absolute paths to imu_pkl files that contain the raw IMU data: defaults to 0.
        ----------
        """

        # valid values for booleans
        valid_bools = [0, False, 1, True]

        to_plot = [kwargs['to_plot'] if 'to_plot' in kwargs.keys() and kwargs['to_plot'] in valid_bools else 0][0]
        npx_sampling_rate = float([kwargs['npx_sampling_rate'] if 'npx_sampling_rate' in kwargs.keys() else 3e4][0])
        ground_probe = int([kwargs['ground_probe'] if 'ground_probe' in kwargs.keys() else 0][0])
        imu_files = [kwargs['imu_files'] if 'imu_files' in kwargs.keys() and type(kwargs['imu_files']) == list else 0][0]

        # save the sync and IMU data in a dictionary where file names are keys
        sync_data = {}
        teensy_times = {}
        
        for fileind, pkl_file in enumerate(self.sync_pkls):
            if os.path.exists(pkl_file):

                with open(pkl_file, 'rb') as one_file:
                    sync_data[pkl_file] = pickle.load(one_file)

                if imu_files != 0:
                    if os.path.exists(imu_files[fileind]):

                        with open(imu_files[fileind], 'rb') as imu_file:
                            temp_file = pickle.load(imu_file)

                        sample_array = temp_file['sample.time (ms)'].tolist()

                        diffs = np.zeros(len(sample_array) - 1)
                        for ind, item in enumerate(sample_array):
                            if 0 < ind < len(sample_array)-1:
                                diffs[ind] = item - sample_array[ind - 1]

                        teensy_times[pkl_file] = diffs

                    else:
                        print('File: {} does not exist!'.format(imu_files[fileind]))
                        sys.exit()

            else:
                print('File: {} does not exist!'.format(pkl_file))
                sys.exit()

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
                elif 'tracking' in data_stream or 'teensy' in data_stream:
                    other_data_cols.append(data_stream)

            for data_stream in other_data_cols:

                data_stream_dict = {}

                for imec_data in imec_data_cols:

                    # reduce the dataframe to only include LED occurrences
                    imec_data_col = temp_df.columns.tolist().index(imec_data)
                    data_stream_col = temp_df.columns.tolist().index(data_stream)
                    reduced_df = temp_df.iloc[2:-2, [imec_data_col, data_stream_col]]
                    extra_data = 0

                    # transform data for imec (to seconds) regardless of the data stream (tracking/imu) and add extra_data to be predicted if necessary
                    if data_stream == 'tracking':
                        # convert imec sample numbers into seconds
                        reduced_df.iloc[:, 0] = reduced_df.iloc[:, 0] / npx_sampling_rate
                        extra_data = np.ravel(temp_df.loc['TTL input start', 'tracking'])
                    else:
                        # convert imec sample and teensy clock numbers into seconds
                        reduced_df.iloc[:, 0] = reduced_df.iloc[:, 0] / npx_sampling_rate
                        reduced_df.iloc[:, 1] = reduced_df.iloc[:, 1] / 1e3

                    # regress
                    regress_class = regress.LinRegression(reduced_df)
                    data_stream_dict[imec_data] = regress_class.split_train_test_and_regress(xy_order=[1, 0], extra_data=extra_data)

                # print relevant information and save empirical frame rate in the last column of .pkl file
                differences_dict = {}
                if data_stream == 'tracking':
                    empirical_frame_rates = []
                    for imec_data in imec_data_cols:

                        print('For imec probe {} sample time predicted by the Motive frames:'.format(int(imec_data[-1])))
                        true_predicted_differences = (data_stream_dict[imec_data]['y_test'] - data_stream_dict[imec_data]['y_test_predictions'])*1e3
                        differences_dict[imec_data] = true_predicted_differences

                        print('The offset between the predicted start of tracking and the TTL start tracking pulse is {:.2f} ms.'.format(np.abs((temp_df.loc['TTL input start', imec_data] / (npx_sampling_rate/1e3)) - (data_stream_dict[imec_data]['extra_data_predictions'][0]*1e3))))
                        print('The differences between NPX test and NPX test predictions are: median {:.2f} ms, mean {:.2f} ms, max {:.2f} ms.'.format(np.abs(np.nanmedian(true_predicted_differences)), np.abs(np.nanmean(true_predicted_differences)), np.nanmax(np.abs(true_predicted_differences))))

                        emp_fr_rate = round(1/data_stream_dict[imec_data]['slope'], 5)
                        empirical_frame_rates.append(emp_fr_rate)
                        print('The estimated empirical frame rate is {:.5f}.'.format(emp_fr_rate))

                    # re-save .pkl file with new frame rate
                    temp2_df = temp_df.copy()
                    temp2_df['Est true fps'] = np.zeros(temp2_df.shape[0])
                    temp2_df.iloc[0, -1] = np.round(empirical_frame_rates[ground_probe], 5)
                    print('Saving empirical capture rate: {:.5f} fps.'.format(empirical_frame_rates[ground_probe]))

                    with open('{}'.format(file_key), 'wb') as df:
                        pickle.dump(temp2_df, df)

                else:
                    for imec_data in imec_data_cols:

                        true_predicted_differences = (data_stream_dict[imec_data]['y_test'] - data_stream_dict[imec_data]['y_test_predictions'])*1e3
                        differences_dict[imec_data] = true_predicted_differences

                        print('For imec probe {} sample time predicted by the Teensy clock:'.format(int(imec_data[-1])))
                        print('The differences between NPX test and NPX test predictions are: median {:.2f} ms, mean {:.2f} ms, max {:.2f} ms.'.format(np.abs(np.nanmedian(true_predicted_differences)), np.abs(np.nanmean(true_predicted_differences)), np.nanmax(np.abs(true_predicted_differences))))

                # plot NPX test and NPX test prediction differences
                if to_plot:
                    colors = {'imec0': '#EE5C42', 'imec1': '#1E90FF'}
                    fig = make_subplots(rows=len(data_stream_dict.keys()), cols=2)
                    fig.update_layout(height=500*len(data_stream_dict.keys()), width=1000, plot_bgcolor='#FFFFFF', title='{}'.format(data_stream), showlegend=True)
                    fig_order = [[1, 2], [3, 4]]
                    for indx, imec_data in enumerate(imec_data_cols):
                        fig.append_trace(go.Scatter(x=data_stream_dict[imec_data]['y_test'], y=data_stream_dict[imec_data]['y_test_predictions'], mode='markers', name='{} predictions'.format(imec_data), marker=dict(color=colors[imec_data], size=5)), row=indx+1, col=1)
                        fig['layout']['xaxis{}'.format(fig_order[indx][0])].update(title='NPX test (s)')
                        fig['layout']['yaxis{}'.format(fig_order[indx][0])].update(title='NPX test predictions (s)')

                        fig.append_trace(go.Histogram(x=differences_dict[imec_data], nbinsx=15, name='{} errors'.format(imec_data), marker_color=colors[imec_data], opacity=.75), row=indx+1, col=2)
                        fig['layout']['xaxis{}'.format(fig_order[indx][1])].update(title='true - predicted (ms)')
                        fig['layout']['yaxis{}'.format(fig_order[indx][1])].update(title='count')
                    fig.show()

                    if data_stream != 'tracking' and imu_files != 0:
                        # plot the distribution of time differences between successive Teensy samples
                        print('The minimum difference between subsequent Teensy samples is {} ms, while the maximum is {} ms.'.format(np.min(teensy_times[file_key]), np.max(teensy_times[file_key])))
                        fig2 = go.Figure()
                        fig2.update_layout(height=500, width=500, yaxis_type='log', plot_bgcolor='#FFFFFF', showlegend=False)
                        fig2.add_trace(go.Histogram(x=teensy_times[file_key], nbinsx=10, marker_color='#458B74', opacity=.75))
                        fig2['layout']['xaxis1'].update(title='Teensy sampling rate variations (ms)')
                        fig2['layout']['yaxis1'].update(title='count')
                        fig2.show()
