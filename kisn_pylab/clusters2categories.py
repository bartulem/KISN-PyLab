# -*- coding: utf-8 -*-

"""

@author: bartulem

Compute cluster quality measures for 'good' and 'mua' clusters.

This script computes various cluster quality measures for clusters
labeled as 'good' or 'mua' in the cluster_info.tsv file. Measures include
various waveform metrics (SNR, spike duration, FWHM, PT-ratio), cluster
isolation metrics (Mahalanobis distance, nearest neighbor and LDA distances)
and ISI violation rates (with functions largely inherited from the Allen Institute
GitHub repository).

The results of the computations are stored in a separate .json file in the
same directory as the Kilosort2 results, and if desirable, one can set a
criterion to separate 'good' units from 'mua' such that the cluster_info.tsv
and cluster_group.tsv files are changed accordingly.

"""

import os
import sys
import gc
import numpy as np
import pandas as pd
import json
import io
import time
from tqdm.notebook import tqdm
from scipy.spatial.distance import cdist
from scipy.stats import chi2
from sklearn.neighbors import NearestNeighbors
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import pickle
import warnings

warnings.simplefilter('ignore')

np.random.seed(1000)


class ClusterQuality:

    # initializer / instance attributes
    def __init__(self, kilosort_output_dir):
        self.kilosort_output_dir = kilosort_output_dir

    def get_waveforms(self, input_dict):

        """
        Gets spike waveform features.

        Adapted from https://github.com/AllenInstitute/ecephys_spike_sorting

        Inputs
        ----------
        input_dict : dictionary
            unit_id : int
                ID for this unit; must be given.
            peak_ch : int
                The peak waveform channel for this unit, must be given.
            spikes : np.ndarray
                Spike times (in samples); must be given.
            spikes_per_rec : int
                Max number of spikes per recording; defaults to 1000.
            samples_per_spike : int
                Number of samples to extract for each spike; defaults to 82.
            bit_volts : float
                NPX has a resolution of 2.3438 µV/bit; defaults to 2.3438.
            pre_samples : int
                Number of samples prior to peak; defaults to 20.
            save_waveforms : boolean (0/False or 1/True)
                Save waveform data as a .npy file; defaults to 0.
        ----------

        Outputs
        ----------
        output : dictionary
            'snr' : float
                Ratio between peak-through span and the waveform SD.
            'waveform_duration' : float
                The time difference between peak and through (in ms).
            'fwhm' : float
                The full spike width at half maximum (in ms).
            'pt_ratio' : float
                The ratio between peak and through.
        ----------
        """

        unit_id = input_dict['unit_id']
        peak_ch = input_dict['peak_ch']
        spikes = input_dict['spikes']
        spikes_per_rec = input_dict['spikes_per_rec'] if 'spikes_per_rec' in input_dict.keys() else 1000
        samples_per_spike = input_dict['samples_per_spike'] if 'samples_per_spike' in input_dict.keys() else 82
        bit_volts = input_dict['bit_volts'] if 'bit_volts' in input_dict.keys() else 2.3438
        pre_samples = input_dict['pre_samples'] if 'pre_samples' in input_dict.keys() else 20
        save_waveforms = input_dict['save_waveforms'] if 'save_waveforms' in input_dict.keys() else 0

        # create empty array (num of spikes considered X num of chans X num of samples)
        waveforms = np.empty((spikes_per_rec, self.raw_data.shape[0], samples_per_spike))
        waveforms[:] = np.nan

        # select only spikes that happen within bounds defined by samples_per_spike
        spikes_selected = spikes[np.logical_and(spikes >= pre_samples, spikes
                                                <= (self.raw_data.shape[1] - (samples_per_spike - pre_samples)))]

        np.random.shuffle(spikes_selected)

        for wv_idx, peak_time in enumerate(spikes_selected[:spikes_per_rec]):
            start = int(peak_time - pre_samples)
            end = start + samples_per_spike

            waveforms[wv_idx, :, :] = self.raw_data[:, start:end] * bit_volts

        if save_waveforms:
            np.save('{}{}waveforms{}cl{:04d}'.format(self.kilosort_output_dir,
                                                     os.sep, os.sep,
                                                     unit_id), waveforms)

        # get mean waveform at the peak channel
        mean_waveform = np.nanmean(waveforms[:, peak_ch, :], axis=0)

        # # # get SNR
        peak_trough_span = np.max(mean_waveform) - np.min(mean_waveform)
        waveform_error = waveforms[:, peak_ch, :] - np.tile(mean_waveform, (np.shape(waveforms[:, peak_ch, :])[0], 1))
        snr = peak_trough_span / (2 * np.nanstd(waveform_error.flatten()))

        # # # get waveform duration in milliseconds
        trough_indx = np.argmin(mean_waveform)
        peak_indx = np.argmax(mean_waveform)

        # to avoid detecting peak before trough, explained below:
        # sometimes, there could be a peak before the trough and what you want is to
        # get the distance between the trough and the peak that follows it, not the one
        # before - provided that the voltage at the peak is NOT higher than the absolute
        # voltage at the trough. If it is, on the other hand, these could be positive
        # peaked units so you want to get the difference between the first peak and the
        # subsequent trough
        timestamps = np.arange(0, samples_per_spike, 1)
        if mean_waveform[peak_indx] > np.abs(mean_waveform[trough_indx]):
            waveform_duration = timestamps[peak_indx:][np.where(mean_waveform[peak_indx:] == np.min(mean_waveform[peak_indx:]))[0][0]] - timestamps[peak_indx]
        else:
            waveform_duration = timestamps[trough_indx:][np.where(mean_waveform[trough_indx:] == np.max(mean_waveform[trough_indx:]))[0][0]] - timestamps[trough_indx]

        # get FWHM in milliseconds
        try:
            if mean_waveform[peak_indx] > np.abs(mean_waveform[trough_indx]):
                threshold = mean_waveform[peak_indx] * 0.5
                thresh_crossing_1 = np.min(
                    np.where(mean_waveform[:peak_indx] > threshold)[0])
                thresh_crossing_2 = np.min(
                    np.where(mean_waveform[peak_indx:] < threshold)[0]) + peak_indx
            else:
                threshold = mean_waveform[trough_indx] * 0.5
                thresh_crossing_1 = np.min(
                    np.where(mean_waveform[:trough_indx] < threshold)[0])
                thresh_crossing_2 = np.min(
                    np.where(mean_waveform[trough_indx:] > threshold)[0]) + trough_indx

            fwhm = timestamps[thresh_crossing_2] - timestamps[thresh_crossing_1]

        except ValueError:

            fwhm = np.nan

        # get peak-to-trough ratio
        pt_ratio = np.abs(mean_waveform[peak_indx] / mean_waveform[trough_indx])

        return {'snr': snr,
                'waveform_duration': waveform_duration / 30,
                'fwhm': fwhm / 30,
                'pt_ratio': pt_ratio}

    def get_cluster_pcs(self, input_dict):

        """
        Gets PC features for a given cluster.

        Adapted from https://github.com/AllenInstitute/ecephys_spike_sorting

        Inputs
        ----------
        input_dict : dictionary
            unit_id : int
                ID for this unit; must be given.
            subsample : int
                Maximum number of spikes to subsample; must be given.
            channels_to_use : np.ndarray
                Channels around peak channel to use for calculating metrics; must be given.
        ----------

        Outputs
        ----------
        unit_PCs : list
            A list of PC features for a given cluster.
        ----------
        """

        unit_id = input_dict['unit_id']
        subsample = input_dict['subsample']
        channels_to_use = input_dict['channels_to_use']

        inds_for_unit = np.where(self.spike_clusters == unit_id)[0]

        spikes_to_use = np.random.permutation(inds_for_unit)[:subsample]

        unique_template_ids = np.unique(self.spike_templates[spikes_to_use])

        unit_PCs = []

        for template_id in unique_template_ids:

            index_mask = spikes_to_use[np.squeeze(self.spike_templates[spikes_to_use]) == template_id]
            these_inds = self.pc_feature_ind[template_id, :]

            pc_array = []

            for i in channels_to_use:

                if np.isin(i, these_inds):
                    channel_index = np.argwhere(these_inds == i)[0][0]
                    pc_array.append(self.pc_features[index_mask, :, channel_index])
                else:
                    return None

            unit_PCs.append(np.stack(pc_array, axis=-1))

        if len(unit_PCs) > 0:
            return np.concatenate(unit_PCs)
        else:
            return None

    def isi_violations(self, input_dict):

        """
        Computes inter-spike-interval violations.

        Metric described in Hill et al. (2011) J. Neurosci. 31: 8699-8705;
        adapted from https://github.com/AllenInstitute/ecephys_spike_sorting

        Inputs
        ----------
        input_dict : dictionary
            spikes : np.ndarray
                Array of spike times (in seconds); must be given.
            min_isi : int/float
                Minimum threshold (in seconds) for spike duplication; defaults to 0.
            isi_threshold : int/float
                The inter-spike interval (in seconds); defaults to 15e-4.
        ----------

        Outputs
        ----------
        fpRate : float
            The ISI violation rate normalized by the firing rate.
        ----------
        """

        spikes = input_dict['spikes']
        min_isi = input_dict['min_isi'] if 'min_isi' in input_dict.keys() else 0
        isi_threshold = input_dict['isi_threshold'] if 'isi_threshold' in input_dict.keys() else 15e-4

        # remove duplicate spikes
        duplicates = np.where(np.diff(spikes) <= min_isi)[0]
        spikes = np.delete(spikes, duplicates + 1)

        # get total number of ISI violations
        total_violations = np.sum(np.diff(spikes) < isi_threshold)

        # time during which refractory period violations could occur around true spikes
        # the factor of two arises since refractory period violations occur whether
        # a rogue spike appears immediately before or after a true spike
        violation_time = 2 * len(spikes) * (isi_threshold - min_isi)

        # overall firing rate of the cell in the session
        firing_rate = len(spikes) / (self.npx_samples / self.npx_sampling_rate)

        # how many violations happen during the violation time
        violation_rate = total_violations / violation_time

        # violation rate normalized by the firing rate
        fp_rate = violation_rate / firing_rate

        return fp_rate

    def mahalanobis_metrics(self, input_dict):

        """
        Calculates L-ratio and isolation distance in Mahalanobis space.

        Metrics described in Schmitzer-Torbert et al. (2005) Neurosci. 131: 1-11;
        adapted from https://github.com/AllenInstitute/ecephys_spike_sorting

        Inputs:
        -------
        input_dict : dictionary
            unit_id : int
                ID for this unit; must be given.
        ----------

        Outputs
        ----------
        output : dictionary
            'isolation_distance' : float
                Isolation distance of a given unit.
            'l_ratio' : float
                L-ratio for a given unit.
        ----------
        """

        unit_id = input_dict['unit_id']

        pcs_for_this_unit = self.all_pcs[self.all_labels == unit_id, :]
        pcs_for_other_units = self.all_pcs[self.all_labels != unit_id, :]

        mean_value = np.expand_dims(np.mean(pcs_for_this_unit, 0), 0)

        if np.linalg.det(np.cov(pcs_for_this_unit.T)) != 0:
            VI = np.linalg.inv(np.cov(pcs_for_this_unit.T))

        # case of singular matrix
        else:
            return np.nan, np.nan

        mahalanobis_other = np.sort(cdist(mean_value,
                                          pcs_for_other_units,
                                          'mahalanobis', VI=VI)[0])

        mahalanobis_self = np.sort(cdist(mean_value,
                                         pcs_for_this_unit,
                                         'mahalanobis', VI=VI)[0])

        # number of spikes
        n = np.min([pcs_for_this_unit.shape[0], pcs_for_other_units.shape[0]])

        if n >= 2:

            # number of features
            dof = pcs_for_this_unit.shape[1]

            l_ratio = np.sum(1 - chi2.cdf(pow(mahalanobis_other, 2), dof)) / mahalanobis_self.shape[0]
            isolation_distance = pow(mahalanobis_other[n - 1], 2)

        else:
            l_ratio = np.nan
            isolation_distance = np.nan

        return {'isolation_distance': isolation_distance,
                'l_ratio': l_ratio}

    def nearest_neighbors_metrics(self, input_dict):

        """
        Calculates unit contamination based on nearest neighbors search in PC space.

        Metrics described in Chung, et al. (2017) Neuron 95: 1381-1394;
        adapted from https://github.com/AllenInstitute/ecephys_spike_sorting

        Inputs:
        -------
        input_dict : dictionary
            unit_id : int
                ID for this unit; must be given.
            max_spikes_for_nn : int
                Number of spikes to use (calculation can be very slow when this number is >20000); defaults to 10000.
            n_neighbors : int
                Number of neighbors to use; defaults to 4.
        ----------

        Outputs
        ----------
        output : dictionary
            'hit_rate' : float
                Fraction of neighbors for target cluster that are also in target cluster.
            'miss_rate' : float
                Fraction of neighbors outside target cluster that are in target cluster.
        ----------
        """

        unit_id = input_dict['unit_id']
        max_spikes_for_nn = int(input_dict['max_spikes_for_nn'] if 'max_spikes_for_nn' in input_dict.keys() else 10000)
        n_neighbors = int(input_dict['n_neighbors'] if 'n_neighbors' in input_dict.keys() else 4)

        total_spikes = self.all_pcs.shape[0]
        ratio = max_spikes_for_nn / total_spikes
        this_unit = self.all_labels == unit_id

        X = np.concatenate((self.all_pcs[this_unit, :], self.all_pcs[np.invert(this_unit), :]), 0)

        n = np.sum(this_unit)

        if ratio < 1:
            inds = np.arange(0, X.shape[0] - 1, 1 / ratio).astype('int')
            X = X[inds, :]
            n = int(n * ratio)

        nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(X)
        distances, indices = nbrs.kneighbors(X)

        this_cluster_nearest = indices[:n, 1:].flatten()
        other_cluster_nearest = indices[n:, 1:].flatten()

        # fraction of neighbors for target cluster that are also in target cluster
        hit_rate = np.mean(this_cluster_nearest < n)

        # fraction of neighbors outside target cluster that are in target cluster
        miss_rate = np.mean(other_cluster_nearest < n)

        return {'hit_rate': hit_rate,
                'miss_rate': miss_rate}

    def lda_metric(self, input_dict):

        """
        Calculates cluster separation on Fisher's linear discriminant.

        Metric described in Hill et al. (2011) J. Neurosci. 31: 8699-8705;
        adapted from https://github.com/AllenInstitute/ecephys_spike_sorting

        Inputs:
        -------
        input_dict : dictionary
            unit_id : int
                ID for this unit; must be given.
        ----------

        Outputs
        ----------
        d_prime : float
            Isolation distance of a given unit.
        ----------
        """

        unit_id = input_dict['unit_id']

        X = self.all_pcs

        y = np.zeros((X.shape[0],), dtype='bool')
        y[self.all_labels == unit_id] = True

        lda = LDA(n_components=1)

        X_flda = lda.fit_transform(X, y)

        flda_this_cluster = X_flda[np.where(y)[0]]
        flda_other_cluster = X_flda[np.where(np.invert(y))[0]]

        d_prime = (np.mean(flda_this_cluster) - np.mean(flda_other_cluster)) / \
                   np.sqrt(0.5 * (np.std(flda_this_cluster) ** 2 + np.std(flda_other_cluster) ** 2))

        return d_prime

    def compute_quality_measures(self, **kwargs):

        """
        Computes cluster quality measures to separate good units/MUA.

        Inputs
        ----------
        **kwargs: dictionary
        nchan : int/float
            Total number of channels on the NPX probe, for Probe3b should be 385; defaults to 385.
        npx_sampling_rate : int/float
            The sampling rate of the NPX system; defaults to 3e4.
        num_channels_to_compare : int
            The number of channels to look at around peak channel, must be odd; defaults to 13.
        max_spikes_for_cluster : int
            The maximum number of spikes to analyze; defaults to 1000.
        perform_waveforms : boolean (0/False or 1/True)
            Yey or ney on the waveform metrics; defaults to 1.
        perform_isiv : boolean (0/False or 1/True)
            Yey or ney on the isi_violations computation; defaults to 1.
        perform_mahalanobis : boolean (0/False or 1/True)
            Yey or ney on the mahalanobis_metrics; defaults to 1.
        perform_nnm : boolean (0/False or 1/True)
            Yey or ney on the nearest_neighbors_metrics; defaults to 1.
        perform_lda : boolean (0/False or 1/True)
            Yey or ney on the LDA metric; defaults to 1.
        re_categorize : boolean (0/False or 1/True)
            Yey or ney on re-categorizing clusters based on contamination features; defaults to 1.
        save_quality_measures : boolean (0/False or 1/True)
            Yey or ney on saving the cluster_quality_measures.json file; defaults to 1.
        min_spikes : int/float
            The minimum number of spikes a cluster should have to be saved; defaults to 100.
        max_good_isi : int/float
            The maximum acceptable FP ISI metric in order for the cluster to be considered 'good'; defaults to .02.
        max_good_contam : int/float
            The maximum acceptable ContamPct in order for the cluster to be considered 'good'; defaults to 10.
        ----------

        Outputs
        ----------
        cluster_quality_measures : dictionary
            A dictionary with cluster quality measures for each non-noise cluster; saved as .json file.
        cluster_group : pd.DataFrame
            A DataFrame with information about cluster type; overwritten .tsv file.
        cluster_info : pd.DataFrame
            A DataFrame with different information about all clusters; overwritten .tsv file.
        ----------
        """

        # valid values for booleans
        valid_booleans = [0, False, 1, True]

        self.nchan = int(kwargs['nchan'] if 'nchan' in kwargs.keys() and (type(kwargs['nchan']) == int or type(kwargs['nchan']) == float) else 385)
        self.npx_sampling_rate = int(kwargs['npx_sampling_rate'] if 'npx_sampling_rate' in kwargs.keys() else 3e4)
        perform_isiv = kwargs['perform_isiv'] if 'perform_isiv' in kwargs.keys() and kwargs['perform_isiv'] in valid_booleans else 1
        perform_mahalanobis = kwargs['perform_mahalanobis'] if 'perform_mahalanobis' in kwargs.keys() and kwargs['perform_mahalanobis'] in valid_booleans else 1
        perform_nnm = kwargs['perform_nnm'] if 'perform_nnm' in kwargs.keys() and kwargs['perform_nnm'] in valid_booleans else 1
        perform_lda = kwargs['perform_lda'] if 'perform_lda' in kwargs.keys() and kwargs['perform_lda'] in valid_booleans else 1
        perform_waveforms = kwargs['perform_waveforms'] if 'perform_waveforms' in kwargs.keys() and kwargs['perform_waveforms'] in valid_booleans else 1
        num_channels_to_compare = int(kwargs['num_channels_to_compare'] if 'num_channels_to_compare' in kwargs.keys() else 13)
        max_spikes_for_cluster = int(kwargs['max_spikes_for_cluster'] if 'max_spikes_for_cluster' in kwargs.keys() else 1000)
        re_categorize = kwargs['re_categorize'] if 're_categorize' in kwargs.keys() and kwargs['re_categorize'] in valid_booleans else 1
        save_quality_measures = kwargs['save_quality_measures'] if 'save_quality_measures' in kwargs.keys() and kwargs['save_quality_measures'] in valid_booleans else 1
        min_spikes = kwargs['min_spikes'] if 'min_spikes' in kwargs.keys() and (type(kwargs['min_spikes']) == int or type(kwargs['min_spikes']) == float) else 100
        max_good_isi = kwargs['max_good_isi'] if 'max_good_isi' in kwargs.keys() and (type(kwargs['max_good_isi']) == int or type(kwargs['max_good_isi']) == float) else 2e-2
        max_good_contam = kwargs['max_good_contam'] if 'max_good_contam' in kwargs.keys() and (type(kwargs['max_good_contam']) == int or type(kwargs['max_good_contam']) == float) else 10

        # check that the directory is there
        if not os.path.exists(self.kilosort_output_dir):
            print('Could not find directory {}, try again.'.format(self.kilosort_output_dir))
            sys.exit()
        else:
            print('Working on directory {}, please be patient - this could take several hours.'.format(self.kilosort_output_dir))

        # keep time
        start_time = time.time()

        # read the total number of sessions from the concatenation .pkl
        # if it's not there, there was only one recording session
        for file in os.listdir(self.kilosort_output_dir):
            if file.endswith('.pkl'):
                with open('{}{}{}'.format(self.kilosort_output_dir, os.sep, file), 'rb') as pkl_len:
                    session_number = len(pickle.load(pkl_len).keys()) - 1
                break
        else:
            session_number = 1

        if os.path.exists('{}{}cluster_info.tsv'.format(self.kilosort_output_dir, os.sep)):

            # get length of recording in samples
            for afile in os.listdir(self.kilosort_output_dir):
                if 'bin' in afile:
                    npx_recording = np.memmap('{}{}{}'.format(self.kilosort_output_dir, os.sep, afile),
                                              mode='r', dtype=np.int16, order='C')
                    self.npx_samples = npx_recording.shape[0] // self.nchan

                    # reshape the array such that channels are rows and samples are columns
                    self.raw_data = npx_recording.reshape((self.nchan, self.npx_samples), order='F')

                    # delete the map object from memory
                    del npx_recording
                    gc.collect()

                    break
            else:
                print('Could not find raw .bin file in this directory, try again.')
                sys.exit()

            # info about cluster number and cluster type
            self.cluster_df_reduced = pd.read_csv('{}{}cluster_group.tsv'.format(self.kilosort_output_dir, os.sep), sep='\t')

            # info about all clusters
            self.cluster_df = pd.read_csv('{}{}cluster_info.tsv'.format(self.kilosort_output_dir, os.sep), sep='\t')

            # cluster IDs of all the spikes
            self.spike_clusters = np.load('{}{}spike_clusters.npy'.format(self.kilosort_output_dir, os.sep))

            # spike times of all the clusters
            self.spike_times = np.load('{}{}spike_times.npy'.format(self.kilosort_output_dir, os.sep))

            # pc features inds
            self.pc_feature_ind = np.load('{}{}pc_feature_ind.npy'.format(self.kilosort_output_dir, os.sep))

            # pc features
            self.pc_features = np.load('{}{}pc_features.npy'.format(self.kilosort_output_dir, os.sep))

            # spike templates
            self.spike_templates = np.load('{}{}spike_templates.npy'.format(self.kilosort_output_dir, os.sep))

        else:
            print('Could not find cluster_info.tsv file, create it by modifying a cluster ID in Phy.')
            sys.exit()

        # create dictionary where all data is stored
        cluster_quality_dictionary = {}

        # prepare dara for cluster PCs
        if perform_mahalanobis or perform_nnm or perform_lda:

            cluster_ids = np.unique(self.spike_clusters)
            template_ids = np.unique(self.spike_templates)

            template_peak_channels = np.zeros((len(template_ids),), dtype='uint16')
            cluster_peak_channels = np.zeros((len(cluster_ids),), dtype='uint16')

            for tidx, template_id in enumerate(template_ids):
                for_template = np.squeeze(self.spike_templates == template_id)
                pc_max = np.argmax(np.mean(self.pc_features[for_template, 0, :], 0))
                template_peak_channels[tidx] = self.pc_feature_ind[template_id, pc_max]

            for cidx, cluster_id in enumerate(cluster_ids):
                for_unit = np.squeeze(self.spike_clusters == cluster_id)
                templates_for_unit = np.unique(self.spike_templates[for_unit])
                template_positions = np.where(np.isin(template_ids, templates_for_unit))[0]
                cluster_peak_channels[cidx] = np.median(template_peak_channels[template_positions])

            half_spread = int((num_channels_to_compare - 1) / 2)

        # convert spike sample times to seconds
        for idx, unit_id in enumerate(tqdm(self.cluster_df.loc[:, 'id'])):

            # *ContamPct* comes from Kilosort2 and is the ratio of the event rate in the central 2ms bin
            # of the histogram to the baseline of the auto-correlogram (the "shoulders"). It is an
            # estimate of how contaminated a unit is with spikes from other units. This is computed based
            # on the size of the dip in the auto-correlogram compared to what you'd expect from Poisson
            # background contamination.

            # The template *amp* is a less noisy estimate of the size of a spike, because it measures
            # the match of the spike to the entire spatiotemporal waveform of that neuron. This is also
            # the quantity that Kilosort2 thresholds to find the spikes in the first place.

            cluster_quality_dictionary[unit_id] = {'KS_label': self.cluster_df.loc[idx, 'KSLabel'],
                                                   'Phy_label': self.cluster_df.loc[idx, 'group'],
                                                   'isi_violations': np.nan,
                                                   'mahalanobis_metrics': {'isolation_distance': np.nan,
                                                                           'l_ratio': np.nan},
                                                   'nn_metrics': {'hit_rate': np.nan,
                                                                  'miss_rate': np.nan},
                                                   'lda_metric': np.nan,
                                                   'ContamPct': self.cluster_df.loc[idx, 'ContamPct'],
                                                   'Amplitude': self.cluster_df.loc[idx, 'Amplitude'],
                                                   'amp': self.cluster_df.loc[idx, 'amp'],
                                                   'waveform_metrics': {'snr': np.nan,
                                                                        'waveform_duration': np.nan,
                                                                        'fwhm': np.nan,
                                                                        'pt_ratio': np.nan},
                                                   'new_label': 'noise'}

            if self.cluster_df.loc[idx, 'group'] != 'noise':

                # get spiking array in seconds
                cluster_indices = np.where(self.spike_clusters == unit_id)[0]
                spikes = np.take(self.spike_times, cluster_indices)
                spikes_seconds = spikes / self.npx_sampling_rate

                if perform_waveforms:
                    cluster_quality_dictionary[unit_id]['waveform_metrics'] = \
                        ClusterQuality.get_waveforms(self, input_dict={'spikes': spikes,
                                                                       'unit_id': unit_id,
                                                                       'peak_ch': self.cluster_df.loc[idx, 'ch']})

                if perform_isiv:
                    cluster_quality_dictionary[unit_id]['isi_violations'] = \
                        ClusterQuality.isi_violations(self, input_dict={'spikes': spikes_seconds})

                if perform_mahalanobis or perform_nnm or perform_lda:

                    peak_channel = cluster_peak_channels[idx]
                    num_spikes_in_cluster = np.sum(self.spike_clusters == unit_id)

                    half_spread_down = peak_channel \
                        if peak_channel < half_spread \
                        else half_spread

                    half_spread_up = np.max(self.pc_feature_ind) - peak_channel \
                        if peak_channel + half_spread > np.max(self.pc_feature_ind) \
                        else half_spread

                    channels_to_use = np.arange(peak_channel - half_spread_down, peak_channel + half_spread_up + 1)

                    units_in_range = cluster_ids[np.isin(cluster_peak_channels, channels_to_use)]

                    spike_counts = np.zeros(units_in_range.shape)

                    for idx2, cluster_id2 in enumerate(units_in_range):
                        spike_counts[idx2] = np.sum(self.spike_clusters == cluster_id2)

                    if num_spikes_in_cluster > max_spikes_for_cluster:
                        relative_counts = spike_counts / num_spikes_in_cluster * max_spikes_for_cluster
                    else:
                        relative_counts = spike_counts

                    all_pcs = np.zeros((0, self.pc_features.shape[1], channels_to_use.size))
                    all_labels = np.zeros((0,))

                    for idx2, cluster_id2 in enumerate(units_in_range):

                        subsample = int(relative_counts[idx2])

                        pcs = ClusterQuality.get_cluster_pcs(self, input_dict={'unit_id': unit_id,
                                                                               'channels_to_use': channels_to_use,
                                                                               'subsample': subsample})

                        if pcs is not None and len(pcs.shape) == 3:
                            labels = np.ones((pcs.shape[0],)) * cluster_id2

                            all_pcs = np.concatenate((all_pcs, pcs), 0)
                            all_labels = np.concatenate((all_labels, labels), 0)

                    all_pcs = np.reshape(all_pcs, (all_pcs.shape[0], self.pc_features.shape[1] * channels_to_use.size))

                    self.all_pcs = all_pcs
                    self.all_labels = all_labels

                    if perform_mahalanobis:
                        if self.all_pcs.shape[0] > 10:
                            cluster_quality_dictionary[unit_id]['mahalanobis_metrics'] = ClusterQuality.mahalanobis_metrics(self, input_dict={'unit_id': unit_id})

                    if perform_nnm:
                        if self.all_pcs.shape[0] > 10:
                            cluster_quality_dictionary[unit_id]['nn_metrics'] = ClusterQuality.nearest_neighbors_metrics(self, input_dict={'unit_id': unit_id})

                    if perform_lda:
                        if self.all_pcs.shape[0] > 10:
                            cluster_quality_dictionary[unit_id]['lda_metric'] = ClusterQuality.lda_metric(self, input_dict={'unit_id': unit_id})

                # # # re-label clusters according to contamination measures, sequence of steps below:
                # (1) check that the total number of spikes in the cluster is bigger than/equal to session_number*min_spikes
                # (2) to separate good units from MUA we check whether the isi_violations parameter
                #     OR the ContamPct are smaller than the minimum set values.

                if re_categorize:
                    if self.cluster_df.loc[idx, 'n_spikes'] >= session_number * min_spikes:
                        if cluster_quality_dictionary[unit_id]['isi_violations'] < max_good_isi or cluster_quality_dictionary[unit_id]['ContamPct'] < max_good_contam:
                            cluster_quality_dictionary[unit_id]['new_label'] = 'good'
                            self.cluster_df.loc[idx, 'group'] = 'good'
                            self.cluster_df_reduced.loc[idx, 'group'] = 'good'
                        else:
                            cluster_quality_dictionary[unit_id]['new_label'] = 'mua'
                            self.cluster_df.loc[idx, 'group'] = 'mua'
                            self.cluster_df_reduced.loc[idx, 'group'] = 'mua'
                    else:
                        cluster_quality_dictionary[unit_id]['new_label'] = 'noise'
                        self.cluster_df.loc[idx, 'group'] = 'noise'
                        self.cluster_df_reduced.loc[idx, 'group'] = 'noise'

        # save cluster_quality_dictionary & over-write existing .tsv files
        if save_quality_measures:
            with io.open('{}{}cluster_quality_measures.json'.format(self.kilosort_output_dir, os.sep), 'w', encoding='utf-8') as to_save_file:
                to_save_file.write(json.dumps(cluster_quality_dictionary, ensure_ascii=False, indent=4))

        if re_categorize:
            self.cluster_df_reduced.to_csv('{}{}cluster_group.tsv'.format(self.kilosort_output_dir, os.sep), index=False, sep='\t')
            self.cluster_df.to_csv('{}{}cluster_info.tsv'.format(self.kilosort_output_dir, os.sep), index=False, sep='\t')

        print('Cluster quality analysis complete! It took {:.2f} hours.\n'.format((time.time() - start_time) / 3600))
