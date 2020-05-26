# -*- coding: utf-8 -*-

"""

@author: bartulem

Get spike times for individual sessions by splitting clusters.

After cluster-cutting, regardless of whether the sessions were concatenated or not, the spike times
for every individual cluster need to be converted from sample number into seconds, relative to session
start. Matters are more complicated in the multi-probe configuration, as one sample stream needs to be
re-calculated relative to the other (i.e. ground probe) sample stream (because they are running on
different clocks). This is done by regressing one stream onto another and using the fit model to predict
what the spike sample occurrences would be if recorded on the ground probe clock.

"""

import pandas as pd
import numpy as np
import os
import sys
import time
import scipy.io as sio
import pickle
from kisn_pylab import regress
from plotly.subplots import make_subplots
import plotly.graph_objects as go


class ExtractSpikes:

    # initializer / instance attributes
    def __init__(self, the_dirs):
        self.the_dirs = the_dirs

    def split_clusters(self, **kwargs):

        """
        Parameters
        ----------
        **kwargs: dictionary
        one_session : boolean (0/False or 1/True)
            If False - splits multiple sessions, otherwise - converts spike times of one session to seconds; defaults to True.
        min_spikes : int/float
            The minimum number of spikes a cluster should have to be saved; defaults to 100.
        nchan : int/float
            Total number of channels on the NPX probe, for Probe3b should be 385; defaults to 385.
        npx_sampling_rate : int/float
            The sampling rate of the NPX system; defaults to 3e4.
        pkl_lengths : list
            .pkl file that has information about where concatenated files were stitched together; defaults to 0.
        sync_pkls : list
            List which contains paths to as many sync .pkl files as there are recording sessions; defaults to 0.
        ground_probe : int
            In a multi probe setting, the probe other probes are synced to; defaults to 0.
        to_plot : boolean (0/False or 1/True)
            To plot or not to plot y_test and y_test_prediction scatter plot; defaults to 0.
        print_details : boolean (0/False or 1/True)
            Whether or not to print details about spikes in every individual cluster; defaults to 0.
        ----------
        """

        # check that all the directories are there
        for one_dir in self.the_dirs:
            if not os.path.exists(one_dir):
                print('Could not find directory {}, try again.'.format(one_dir))
                sys.exit()

        # valid values for booleans
        valid_bools = [0, False, 1, True]

        one_session = [kwargs['one_session'] if 'one_session' in kwargs.keys() and kwargs['one_session'] in valid_bools else 1][0]
        nchan = int([kwargs['nchan'] if 'nchan' in kwargs.keys() and (type(kwargs['nchan']) == int or type(kwargs['nchan']) == float) else 385][0])
        min_spikes = [kwargs['min_spikes'] if 'min_spikes' in kwargs.keys() and (type(kwargs['min_spikes']) == int or type(kwargs['min_spikes']) == float) else 100][0]
        npx_sampling_rate = float([kwargs['npx_sampling_rate'] if 'npx_sampling_rate' in kwargs.keys() else 3e4][0])
        pkl_lengths = [kwargs['pkl_lengths'] if 'pkl_lengths' in kwargs.keys() else 0][0]
        sync_pkls = [kwargs['sync_pkls'] if 'sync_pkls' in kwargs.keys() else 0][0]
        ground_probe = int([kwargs['ground_probe'] if 'ground_probe' in kwargs.keys() else 0][0])
        to_plot = [kwargs['to_plot'] if 'to_plot' in kwargs.keys() and kwargs['to_plot'] in valid_bools else 0][0]
        print_details = [kwargs['print_details'] if 'print_details' in kwargs.keys() and kwargs['print_details'] in valid_bools else 0][0]

        # check that the .pkl concatenation files are there
        if pkl_lengths != 0:
            for pkl_len in pkl_lengths:
                if not os.path.exists(pkl_len):
                    print('Could not find file {}, try again.'.format(pkl_len))
                    sys.exit()

        # check that the sync .pkl files are there
        if sync_pkls != 0:
            for sync_pkl in sync_pkls:
                if not os.path.exists(sync_pkl):
                    print('Could not find file {}, try again.'.format(sync_pkl))
                    sys.exit()

        # load the appropriate spike data
        spike_dict = {}
        for dirind, one_dir in enumerate(self.the_dirs):

            dir_key = 'imec{}'.format(one_dir[one_dir.find('imec') + len('imec')])
            spike_dict[dir_key] = {}

            # info about all clusters
            spike_dict[dir_key]['cluster_info'] = pd.read_csv('{}{}cluster_info.tsv'.format(one_dir, os.sep), sep='\t')

            # cluster IDs of all the spikes
            spike_dict[dir_key]['spike_clusters'] = np.load('{}{}spike_clusters.npy'.format(one_dir, os.sep))

            # spike times of all the clusters
            spike_dict[dir_key]['spike_times'] = np.load('{}{}spike_times.npy'.format(one_dir, os.sep))

            # load the .pkl file with total file lengths
            with open(pkl_lengths[dirind], 'rb') as pkl_len:
                spike_dict[dir_key]['file_lengths'] = pickle.load(pkl_len)

            # keep info about the directory
            spike_dict[dir_key]['dir'] = one_dir

        t = time.time()
        print('Splitting clusters to individual sessions, please be patient - this could take awhile (depending on the number of clusters).')

        # check if one or more sessions require splitting
        if not one_session:

            # get spikes from every good cluster and save them as spike times according to each session start
            for probe in spike_dict.keys():

                probe_id = int(probe[-1])

                # create a big dictionary where all the cells go
                probe_spike_data = {}

                # extract cluster data
                cluster_info = spike_dict[probe]['cluster_info']
                spike_clusters = spike_dict[probe]['spike_clusters']
                spike_times = spike_dict[probe]['spike_times']

                # load changepoints for different sessions
                file_lengths = spike_dict[probe]['file_lengths']

                for indx in range(cluster_info.shape[0]):
                    if cluster_info.loc[indx, 'group'] == 'good':
                        cluster_indices = np.where(spike_clusters == cluster_info.loc[indx, 'id'])[0]
                        spikes_all_sessions = np.take(spike_times, cluster_indices)

                        # creat spike_dict and put each spike in the appropriate session
                        spikes_dict = {'session_{}'.format(x + 1): [] for x in range(len(file_lengths.keys()) - 1)}
                        for aspike in spikes_all_sessions:
                            truth = True
                            while truth:
                                for xx in range(len(file_lengths.keys()) - 1):
                                    lower_bound = file_lengths['total_len_changepoints'][xx] // nchan
                                    upper_bound = file_lengths['total_len_changepoints'][xx + 1] // nchan
                                    if lower_bound <= aspike < upper_bound:
                                        if probe_id == ground_probe:
                                            spikes_dict['session_{}'.format(xx + 1)].append((aspike - lower_bound) / npx_sampling_rate)
                                        else:
                                            spikes_dict['session_{}'.format(xx + 1)].append(aspike - lower_bound)
                                        truth = False

                            if not probe_id == ground_probe:
                                for sessionindx, asession in enumerate(spikes_dict.keys()):

                                    # read in sync .pkl file (there should only be one in the list for one recording session)
                                    with open(sync_pkls[sessionindx], 'rb') as sync_pkl:
                                        full_session_df = pickle.load(sync_pkl)

                                    # separate the imec part of the dataframe
                                    reduced_session_df = full_session_df.iloc[1:-1, :2]

                                    # regress
                                    regress_session_class = regress.LinRegression(reduced_session_df)
                                    regression_session_dict = regress_session_class.split_train_test_and_regress(xy_order=[probe_id, 1-probe_id], extra_data=np.array(spikes_dict['session_{}'.format(sessionindx + 1)]))

                                    # prepare arrays for plotting and check if the transformation is acceptable
                                    y_session_test = regression_session_dict['y_test']
                                    y_session_test_predictions = np.array([int(round(sample) for sample in regression_session_dict['y_test_predictions'])])
                                    true_predicted_differences = (y_session_test - y_session_test_predictions) / (npx_sampling_rate / 1e3)
                                    print('The differences between session {} imec test and imec test predictions are: median {:.2f} ms, mean {:.2f} ms, max {:.2f} ms.'.format(sessionindx + 1, np.abs(np.nanmedian(true_predicted_differences)), np.abs(np.nanmean(true_predicted_differences)), np.nanmax(np.abs(true_predicted_differences))))

                                    # plot
                                    if to_plot:
                                        fig = make_subplots(rows=1, cols=2)
                                        fig.update_layout(height=500, width=1000, plot_bgcolor='#FFFFFF', title='imec regression quality', showlegend=True)
                                        fig.append_trace(go.Scatter(x=y_session_test, y=y_session_test_predictions, mode='markers', name='LED test comparison', marker=dict(color='#66CDAA', size=5)), row=1, col=1)
                                        fig['layout']['xaxis1'].update(title='other probe test (samples)')
                                        fig['layout']['yaxis1'].update(title='other probe test predictions (samples)')

                                        fig.append_trace(go.Histogram(x=true_predicted_differences, nbinsx=15, name='LED prediction errors', marker_color='#66CDAA', opacity=.75), row=1, col=2)
                                        fig['layout']['xaxis2'].update(title='true - predicted (ms)')
                                        fig['layout']['yaxis2'].update(title='count')
                                        fig.show()

                                    # allocate spikes
                                    session_spikes = []
                                    for aspike in regression_session_dict['extra_data_predictions']:
                                        if aspike > 0:
                                            session_spikes.append(int(round(aspike)) / npx_sampling_rate)

                                    spikes_dict['session_{}'.format(sessionindx + 1)] = session_spikes

                        if print_details:
                            print('The Phy cluster ID on imec{} is {} (it has a total of {} spikes).'.format(probe_id, cluster_info.loc[indx, 'id'], cluster_info.loc[indx, 'n_spikes']))
                            print('Splitting spikes in {} sessions produced the following results:'.format(len(file_lengths.keys()) - 1))
                            for asession in spikes_dict.keys():
                                print('{} has {} spikes.'.format(asession, len(spikes_dict[asession])))
                            print('In total, {} spikes have been accounted for.'.format(sum([len(spikes_dict[key]) for key in spikes_dict.keys()])))

                        probe_spike_data['imec{}_cell{:03d}_ch{:03d}'.format(probe_id, cluster_info.loc[indx, 'id'], cluster_info.loc[indx, 'ch'])] = spikes_dict

                # save spike .mat files (only if there's more than *min_spikes* spk/session!)
                for session in range(len(file_lengths.keys()) - 1):
                    path = '{}{}imec{}_session{}'.format(spike_dict[probe]['dir'], os.sep, probe_id, session + 1)
                    if not os.path.exists(path):
                        os.makedirs(path)

                    cell_count = 0
                    for acell in probe_spike_data.keys():
                        if len(probe_spike_data[acell]['session_{}'.format(session + 1)]) > min_spikes:
                            sio.savemat(path + os.sep + acell + '.mat', {'cellTS': np.array(probe_spike_data[acell]['session_{}'.format(session + 1)])}, oned_as='column')
                            cell_count += 1
                        else:
                            del probe_spike_data[acell]['session_{}'.format(session + 1)]

                    print('In session {} on imec{}, there are {} good clusters (above {} spikes).'.format(session + 1, probe_id, cell_count, min_spikes))

                # check how many cells were present in all sessions
                omni_present = 0
                all_sessions = ['session_{}'.format(x + 1) for x in range(len(file_lengths.keys()) - 1)]
                for acell in probe_spike_data.keys():
                    if list(probe_spike_data[acell].keys()) == all_sessions:
                        omni_present += 1

                print('On imec{}, {} cells were present in all sessions'.format(probe_id, omni_present))
                print('Processing complete!')

        else:

            # get spikes from every good cluster and save them as spike times according to session start
            for probe in spike_dict.keys():

                probe_id = int(probe[-1])

                # create a dictionary where all the cells go
                probe_spike_data = {}

                # extract cluster data
                cluster_info = spike_dict[probe]['cluster_info']
                spike_clusters = spike_dict[probe]['spike_clusters']
                spike_times = spike_dict[probe]['spike_times']

                for indx in range(cluster_info.shape[0]):
                    if cluster_info.loc[indx, 'group'] == 'good':
                        cluster_indices = np.where(spike_clusters == cluster_info.loc[indx, 'id'])[0]
                        spikes_all = np.take(spike_times, cluster_indices)

                        if print_details:
                            print('The Phy cluster ID is {} (it has a total of {} spikes).'.format(cluster_info.loc[indx, 'id'], cluster_info.loc[indx, 'n_spikes']))

                        spikes = []
                        if probe_id == ground_probe:
                            for aspike in spikes_all:
                                spikes.append(aspike / npx_sampling_rate)
                        else:

                            # read in sync .pkl file (there should only be one in the list for one recording session)
                            with open(sync_pkls[0], 'rb') as sync_pkl:
                                full_session_df = pickle.load(sync_pkl)

                            # separate the imec part of the dataframe
                            reduced_session_df = full_session_df.iloc[1:-1, :2]

                            # regress
                            regress_session_class = regress.LinRegression(reduced_session_df)
                            regression_session_dict = regress_session_class.split_train_test_and_regress(xy_order=[probe_id, 1-probe_id], extra_data=spikes_all)

                            # prepare arrays for plotting and check if the transformation is acceptable
                            y_session_test = regression_session_dict['y_test']
                            y_session_test_predictions = np.array([int(round(sample)) for sample in regression_session_dict['y_test_predictions']])
                            true_predicted_differences = (y_session_test - y_session_test_predictions) / (npx_sampling_rate / 1e3)
                            print('The differences between imec test and imec test predictions are: median {:.2f} ms, mean {:.2f} ms, max {:.2f} ms.'.format(np.abs(np.nanmedian(true_predicted_differences)), np.abs(np.nanmean(true_predicted_differences)), np.nanmax(np.abs(true_predicted_differences))))

                            # plot
                            if to_plot:
                                fig2 = make_subplots(rows=1, cols=2)
                                fig2.update_layout(height=500, width=1000, plot_bgcolor='#FFFFFF', title='imec regression quality', showlegend=True)
                                fig2.append_trace(go.Scatter(x=y_session_test, y=y_session_test_predictions, mode='markers', name='LED test comparison', marker=dict(color='#66CDAA', size=5)), row=1, col=1)
                                fig2['layout']['xaxis1'].update(title='other probe test (samples)')
                                fig2['layout']['yaxis1'].update(title='other probe test predictions (samples)')

                                fig2.append_trace(go.Histogram(x=true_predicted_differences, nbinsx=15, name='LED prediction errors', marker_color='#66CDAA', opacity=.75), row=1, col=2)
                                fig2['layout']['xaxis2'].update(title='true - predicted (ms)')
                                fig2['layout']['yaxis2'].update(title='count')
                                fig2.show()

                            # allocate spikes
                            for aspike in regression_session_dict['extra_data_predictions']:
                                if aspike > 0:
                                    spikes.append(int(round(aspike)) / npx_sampling_rate)

                        probe_spike_data['imec{}_cell{:03d}_ch{:03dg}'.format(probe_id, cluster_info.loc[indx, 'id'], cluster_info.loc[indx, 'ch'])] = spikes

                # save spike .mat files (only if there's more than *min_spikes* spk/session!)
                path = '{}{}imec{}_session1'.format(spike_dict[probe]['dir'], os.sep, probe_id)
                if not os.path.exists(path):
                    os.makedirs(path)

                cell_count = 0
                for acell in probe_spike_data.keys():
                    if len(probe_spike_data[acell]) > min_spikes:
                        sio.savemat(path + os.sep + acell + '.mat', {'cellTS': np.array(probe_spike_data[acell])}, oned_as='column')
                        cell_count += 1

                print('In this session, on imec{} there are {} good clusters (above {} spikes).'.format(probe_id, cell_count, min_spikes))
                print('Processing complete! It took {:.2f} minutes.'.format((time.time() - t) / 60))
