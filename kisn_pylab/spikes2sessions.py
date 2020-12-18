# -*- coding: utf-8 -*-

"""

@author: bartulem

Get spike times for individual sessions by splitting clusters.

After cluster-cutting, regardless of whether the sessions were concatenated or not, the spike times
for every individual cluster need to be converted from sample number into seconds, relative to session
(or tracking) start. Matters are more complicated in the sync-misaligned or multi-probe configurations,
as sample streams have to be re-calculated relative to each other or some master clock.

This script offers two solutions:
(1) assuming a stable NPX sampling frequency and in the multi-probe scenario regressing one sync LED
    stream onto another and using the fit model to predict what the spike sample occurrences would be
    if recorded on the ground probe clock (default sync-aligned condition)
(2) using the time2events.py script to convert each sample spike time to time as measured by the IPI
    generator (optional sync-misaligned condition)

"""

import pandas as pd
import numpy as np
import os
import sys
import time
import json
import io
import scipy.io as sio
import pickle
from kisn_pylab import regress
from kisn_pylab import times2events
from tqdm.notebook import tqdm
from plotly.subplots import make_subplots
import plotly.graph_objects as go


class ExtractSpikes:

    # initializer / instance attributes
    def __init__(self, the_dirs, sync_pkls):
        self.the_dirs = the_dirs
        self.sync_pkls = sync_pkls

    def split_clusters(self, **kwargs):

        """
        Inputs
        ----------
        **kwargs: dictionary
        one_session : boolean (0/False or 1/True)
            If False - splits multiple sessions, otherwise - converts spike times of one session to seconds; defaults to True.
        min_spikes : int/float
            The minimum number of spikes a cluster should have to be saved; defaults to 100.
        nchan : int/float
            Total number of channels on the NPX probe, for probe 3B2 should be 385; defaults to 385.
        npx_sampling_rate : int/float
            The sampling rate of the NPX system; defaults to 3e4.
        pkl_lengths : list
            .pkl file that has information about where concatenated files were stitched together; defaults to 0.
        ground_probe : int
            In a dual probe setting, the probe the other is synced to; defaults to 0.
        to_plot : boolean (0/False or 1/True)
            To plot or not to plot y_test and y_test_prediction scatter plot; defaults to 0.
        print_details : boolean (0/False or 1/True)
            Whether or not to print details about spikes in every individual cluster; defaults to 0.
        important_cluster_groups : list
            The list of relevant cluster groups you want to analyze, should be 'good' and 'mua'; defaults to [good].
        eliminate_duplicates : boolean (0/False or 1/True)
            Whether or not to eliminate duplicate spikes; defaults to 1.
        min_isi : int/float
            Threshold for duplicate spikes in seconds; defaults to 0.
        switch_clock : boolean (0/False or 1/True)
            Convert each sample spike time to time as measured by the IPI generator; defaults to 0.
        ----------

        Outputs
        ----------
        spike times : np.ndarray
            Arrays that contain spike times (in seconds); saved as .mat files in a separate directory.
        cluster_groups_information :
            Information about which cluster belongs to 'good' or 'MUA' categories; saved as .json file.
        ----------
        """

        # check that all the directories are there
        for one_dir in self.the_dirs:
            if not os.path.exists(one_dir):
                print('Could not find directory {}, try again.'.format(one_dir))
                sys.exit()

        # check that the sync .pkl files are there
        for sync_pkl in self.sync_pkls:
            if not os.path.exists(sync_pkl):
                print('Could not find file {}, try again.'.format(sync_pkl))
                sys.exit()

        # valid values for booleans
        valid_booleans = [0, False, 1, True]

        one_session = kwargs['one_session'] if 'one_session' in kwargs.keys() and kwargs['one_session'] in valid_booleans else 1
        nchan = int(kwargs['nchan'] if 'nchan' in kwargs.keys() and (type(kwargs['nchan']) == int or type(kwargs['nchan']) == float) else 385)
        min_spikes = kwargs['min_spikes'] if 'min_spikes' in kwargs.keys() and (type(kwargs['min_spikes']) == int or type(kwargs['min_spikes']) == float) else 100
        npx_sampling_rate = float(kwargs['npx_sampling_rate'] if 'npx_sampling_rate' in kwargs.keys() else 3e4)
        pkl_lengths = kwargs['pkl_lengths'] if 'pkl_lengths' in kwargs.keys() else 0
        ground_probe = int(kwargs['ground_probe'] if 'ground_probe' in kwargs.keys() else 0)
        to_plot = kwargs['to_plot'] if 'to_plot' in kwargs.keys() and kwargs['to_plot'] in valid_booleans else False
        print_details = kwargs['print_details'] if 'print_details' in kwargs.keys() and kwargs['print_details'] in valid_booleans else False
        important_cluster_groups = kwargs['important_cluster_groups'] if 'important_cluster_groups' in kwargs.keys() and type(kwargs['important_cluster_groups']) == list else ['good']
        eliminate_duplicates = kwargs['eliminate_duplicates'] if 'eliminate_duplicates' in kwargs.keys() and kwargs['eliminate_duplicates'] in valid_booleans else True
        min_isi = kwargs['min_isi'] if 'min_isi' in kwargs.keys() and (type(kwargs['min_isi']) == int or type(kwargs['min_isi']) == float) else 0
        switch_clock = kwargs['switch_clock'] if 'switch_clock' in kwargs.keys() and kwargs['switch_clock'] in valid_booleans else False

        # check that the .pkl concatenation files are there
        if pkl_lengths != 0:
            for pkl_len in pkl_lengths:
                if not os.path.exists(pkl_len):
                    print('Could not find file {}, try again.'.format(pkl_len))
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
            if pkl_lengths != 0:
                with open(pkl_lengths[dirind], 'rb') as pkl_len:
                    spike_dict[dir_key]['file_lengths'] = pickle.load(pkl_len)

            # keep info about the directory
            spike_dict[dir_key]['dir'] = one_dir

        start_time = time.time()

        print('Splitting clusters to individual sessions, please be patient - this could '
              'take awhile (depending on the number of sessions and clusters).\n')

        # create dictionary which would store information about cluster groups
        cluster_groups_info = {}

        # get spikes from every good/mua cluster and save them as spike times according to each session/tracking start
        for probe in spike_dict.keys():

            print('Working on {} clusters.\n'.format(probe))

            # each probe gets a place in the cluster groups information dictionary
            cluster_groups_info[probe] = {category: [] for category in important_cluster_groups}

            probe_id = int(probe[-1])

            # create a big dictionary where all the cells go
            probe_spike_data = {}
            probe_spike_frame_data = {}

            # extract cluster data
            cluster_info = spike_dict[probe]['cluster_info']
            spike_clusters = spike_dict[probe]['spike_clusters']
            spike_times = spike_dict[probe]['spike_times']

            if not one_session:
                # load change-points for different sessions
                file_lengths = spike_dict[probe]['file_lengths']

            for idx in tqdm(range(cluster_info.shape[0])):

                # filter only non-noise clusters
                if cluster_info.loc[idx, 'group'] in important_cluster_groups:

                    # get all spikes for that cluster
                    cluster_indices = np.where(spike_clusters == cluster_info.loc[idx, 'id'])[0]
                    spikes_all = np.take(spike_times, cluster_indices)

                    # create spike_dict and put each spike in the appropriate session
                    if one_session:
                        frames_dict = {'session_1': []}
                        spikes_dict = {'session_1': []}
                        for aspike in spikes_all:
                            if switch_clock:
                                spikes_dict['session_1'].append(aspike)
                            else:
                                if probe_id == ground_probe:
                                    spikes_dict['session_1'].append(aspike / npx_sampling_rate)
                                else:
                                    spikes_dict['session_1'].append(aspike)

                    else:
                        frames_dict = {'session_{}'.format(x + 1): [] for x in range(len(file_lengths.keys()) - 1)}
                        spikes_dict = {'session_{}'.format(x + 1): [] for x in range(len(file_lengths.keys()) - 1)}
                        for aspike in spikes_all:
                            for xx in range(len(file_lengths.keys()) - 1):
                                lower_bound = file_lengths['total_len_changepoints'][xx] // nchan
                                upper_bound = file_lengths['total_len_changepoints'][xx + 1] // nchan
                                if lower_bound <= aspike < upper_bound:
                                    if switch_clock:
                                        spikes_dict['session_{}'.format(xx + 1)].append(aspike - lower_bound)
                                    else:
                                        if probe_id == ground_probe:
                                            spikes_dict['session_{}'.format(xx + 1)].append((aspike - lower_bound) / npx_sampling_rate)
                                        else:
                                            spikes_dict['session_{}'.format(xx + 1)].append(aspike - lower_bound)
                                    break

                    for sessionidx, asession in enumerate(spikes_dict.keys()):

                        # read in sync .pkl file (there should only be one in the list for one recording session)
                        with open(self.sync_pkls[sessionidx], 'rb') as sync_pkl:
                            full_session_df = pickle.load(sync_pkl)

                        # put all spikes on sync pulse generator time
                        if switch_clock:
                            spikes_dict['session_{}'.format(sessionidx + 1)], \
                            frames_dict['session_{}'.format(sessionidx + 1)] = times2events.times_to_events(sync_data=full_session_df.to_numpy(dtype=np.float64),
                                                                                                            event_data=np.array(spikes_dict['session_{}'.format(sessionidx + 1)]),
                                                                                                            imec_data_col=full_session_df.columns.tolist().index('imec{}'.format(probe_id)),
                                                                                                            time_data_col=full_session_df.columns.tolist().index('time (ms)'),
                                                                                                            frame_data_col=full_session_df.columns.tolist().index('tracking'))

                        # regress other probe to ground_probe time
                        else:
                            if not probe_id == ground_probe:

                                # separate the imec part of the DataFrame
                                imec0_data_col = full_session_df.columns.tolist().index('imec0')
                                imec1_data_col = full_session_df.columns.tolist().index('imec1')
                                reduced_session_df = full_session_df.iloc[1:-1, [imec0_data_col, imec1_data_col]]

                                # regress
                                regress_session_class = regress.LinRegression(reduced_session_df)
                                regression_session_dict = regress_session_class.split_train_test_and_regress(xy_order=[probe_id, 1 - probe_id],
                                                                                                             extra_data=np.array(spikes_dict['session_{}'.format(sessionidx + 1)]))

                                # prepare arrays for plotting and check if the transformation is acceptable
                                y_session_test = regression_session_dict['y_test']
                                y_session_test_predictions = np.array([int(round(sample)) for sample in regression_session_dict['y_test_predictions']])
                                true_predicted_differences = (y_session_test - y_session_test_predictions) / (npx_sampling_rate / 1e3)
                                print('The differences between session {} imec test and imec test predictions are: '
                                      'median {:.2f} ms, mean {:.2f} ms, max {:.2f} ms.'.format(sessionidx + 1,
                                                                                                np.abs(np.nanmedian(true_predicted_differences)),
                                                                                                np.abs(np.nanmean(true_predicted_differences)),
                                                                                                np.nanmax(np.abs(true_predicted_differences))))

                                # plot
                                if to_plot:
                                    fig = make_subplots(rows=1, cols=2)
                                    fig.update_layout(height=500, width=1000, plot_bgcolor='#FFFFFF', title='imec regression quality', showlegend=True)
                                    fig.append_trace(go.Scatter(x=y_session_test, y=y_session_test_predictions, mode='markers',
                                                                name='LED test comparison', marker=dict(color='#66CDAA', size=5)), row=1, col=1)
                                    fig['layout']['xaxis1'].update(title='other probe test (samples)')
                                    fig['layout']['yaxis1'].update(title='other probe test predictions (samples)')

                                    fig.append_trace(go.Histogram(x=true_predicted_differences, nbinsx=15,
                                                                  name='LED prediction errors', marker_color='#66CDAA', opacity=.75), row=1, col=2)
                                    fig['layout']['xaxis2'].update(title='true - predicted (ms)')
                                    fig['layout']['yaxis2'].update(title='count')
                                    fig.show()

                                # allocate spikes
                                session_spikes = []
                                for aspike in regression_session_dict['extra_data_predictions']:
                                    if aspike > 0:
                                        session_spikes.append(int(round(aspike)) / npx_sampling_rate)

                                spikes_dict['session_{}'.format(sessionidx + 1)] = session_spikes

                    if one_session and print_details:
                        print('The Phy cluster ID is {} (it has a total of {} spikes).'.format(cluster_info.loc[idx, 'id'], cluster_info.loc[idx, 'n_spikes']))

                    if not one_session and print_details:
                        print('The Phy cluster ID on imec{} is {} (it has a total of {} spikes).'.format(probe_id, cluster_info.loc[idx, 'id'], cluster_info.loc[idx, 'n_spikes']))
                        print('Splitting spikes in {} sessions produced the following results:'.format(len(file_lengths.keys()) - 1))
                        for asession in spikes_dict.keys():
                            print('{} has {} spikes.'.format(asession, len(spikes_dict[asession])))
                        print('In total, {} spikes have been accounted for.'.format(sum([len(spikes_dict[key]) for key in spikes_dict.keys()])))

                    # give a unique name to each cluster
                    cluster_id = 'imec{}_cl{:04d}_ch{:03d}'.format(probe_id, cluster_info.loc[idx, 'id'], cluster_info.loc[idx, 'ch'])
                    probe_spike_data[cluster_id] = spikes_dict
                    probe_spike_frame_data[cluster_id] = frames_dict

                    # get cluster_id into the cluster groups information dictionary
                    if cluster_id not in cluster_groups_info[probe][cluster_info.loc[idx, 'group']]:
                        cluster_groups_info[probe][cluster_info.loc[idx, 'group']].append(cluster_id)

            # save spike .mat files (only if there's more than *min_spikes* spk/session!)
            if one_session:
                the_range = 1
            else:
                the_range = len(file_lengths.keys()) - 1

            for session in range(the_range):

                # create directory for spikes if non-existent
                path = '{}{}imec{}_session{}'.format(spike_dict[probe]['dir'], os.sep, probe_id, session + 1)
                if not os.path.exists(path):
                    os.makedirs(path)

                unit_count = 0
                mua_count = 0
                duplicates = {}
                for acell in probe_spike_data.keys():

                    # eliminate duplicates
                    cluster_data = np.array(probe_spike_data[acell]['session_{}'.format(session + 1)])
                    if switch_clock:
                        frame_data = np.array(probe_spike_frame_data[acell]['session_{}'.format(session + 1)])
                    if eliminate_duplicates:
                        duplicate_spikes = np.where(np.diff(cluster_data) <= min_isi)[0]
                        cluster_data = np.delete(cluster_data, duplicate_spikes + 1)
                        if switch_clock:
                            frame_data = np.delete(frame_data, duplicate_spikes + 1)
                        if len(duplicate_spikes) > 0:
                            duplicates[acell] = len(duplicate_spikes)

                    # eliminate clusters below minimum number of spikes
                    if len(cluster_data) > min_spikes:
                        if switch_clock:
                            sio.savemat(path + os.sep + acell + '.mat', {'cellTS': cluster_data, 'cellFS': frame_data}, oned_as='column')
                        else:
                            sio.savemat(path + os.sep + acell + '.mat', {'cellTS': cluster_data}, oned_as='column')
                        if acell in cluster_groups_info[probe]['good']:
                            unit_count += 1
                        elif 'mua' in cluster_groups_info[probe].keys() and acell in cluster_groups_info[probe]['mua']:
                            mua_count += 1
                    else:
                        del probe_spike_data[acell]['session_{}'.format(session + 1)]

                print('In session {} on imec{}, there are {} putative single units and {} MUs (above {} spikes).'.format(session + 1, probe_id, unit_count, mua_count, min_spikes))
                if len(duplicates.keys()) > 0 and print_details:
                    for key in duplicates.keys():
                        print('{} had {} duplicate spikes removed.'.format(key, duplicates[key]))

            # check how many cells were present in all sessions
            if not one_session:
                omni_present_units = 0
                omni_present_mua = 0
                all_sessions = ['session_{}'.format(x + 1) for x in range(len(file_lengths.keys()) - 1)]
                for acell in probe_spike_data.keys():
                    if list(probe_spike_data[acell].keys()) == all_sessions:
                        if acell in cluster_groups_info[probe]['good']:
                            omni_present_units += 1
                        elif 'mua' in cluster_groups_info[probe].keys() and acell in cluster_groups_info[probe]['mua']:
                            omni_present_mua += 1

                print('On imec{}, {} putative single units and {} MUs were present in all sessions.'.format(probe_id, omni_present_units, omni_present_mua))

        # save cluster groups information dictionary to file (it's saved in the imec0 Kilosort results directory)
        if 'mua' in important_cluster_groups:
            with io.open('{}{}cluster_groups_information.json'.format(self.the_dirs[0], os.sep), 'w', encoding='utf-8') as cgi_file:
                cgi_file.write(json.dumps(cluster_groups_info, ensure_ascii=False, indent=4))

        print('Processing complete! It took {:.2f} minute(s).\n'.format((time.time() - start_time) / 60))
