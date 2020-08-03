# -*- coding: utf-8 -*-

"""

@author: bartulem (code origin: Allen Institute GitHub repository)

Estimate surface channel from LFP data.

To acquire an estimate of the surface channel (assuming a part of the probe was not
in the brain), this script relies on the LFP data obtained from the same recording
session as the AP data. After median-centering the raw LFP data and estimating the
power spectral density of each channel, to estimate the brain surface location the
algorithm seeks sharp increases in low-frequency LFP band power.

"""

import sys
import os
import numpy as np
from scipy.signal import welch
from scipy.ndimage.filters import gaussian_filter1d
import matplotlib.pyplot as plt


class SeekSurface:

    def __init__(self, lfp_dir):
        self.lfp_dir = lfp_dir

    def find_surface_channel(self, **kwargs):

        """
        adapted from https://github.com/AllenInstitute/ecephys_spike_sorting

        Inputs
        ----------
        **kwargs: dictionary
        nchan : int/float
            Total number of channels on the NPX probe, for probe 3B2 should be 385; defaults to 385.
        lfp_sampling_frequency : int/float
            The sampling rate of the LPF acquisition; defaults to 2.5e3.
        lfp_gain_setting : int
            The amplifier gain for the LFP band; defaults to 250.
        smoothing_width : int/float
            Gaussian smoothing parameter to reduce channel-to-channel noise; defaults to 5.
        power_threshold : int/float
            Ignore threshold crossings if power is above this level (indicates channels are in the brain); defaults to 2.5.
        diff_threshold : int/float
            Threshold to detect large increases in power at brain surface; defaults to -0.06.
        freq_range : list
            Frequency band for detecting power increases; defaults to [0, 10].
        channel_range : list/boolean
            Channels assumed to be out of brain, but in saline; defaults to 0.
        nfft : int
            Length of FFT used for calculations; defaults to 4096.
        n_passes : int
            Number of times to compute offset and surface channel; defaults to 5.
        skip_s_per_pass : int
            Number of seconds between data chunks used on each pass; defaults to 5.
        max_freq : int
            Maximum frequency to plot; defaults to 150.
        reference_channel : int/boolean(True/False)
            Reference channel on probe (if in use); defaults to False.
        to_plot : boolean (0/False or 1/True)
            Yey or ney on the result figure; defaults to 1.
        colormap : str
            The colormap of choice for the figure; defaults to 'afmhot'.
        save_plot : boolean (0/False or 1/True)
            Yey or ney on saving the figure; defaults to 0.
        fig_format : str
            The format of the saved figure; defaults to 'png'.
        ----------

        Outputs
        ----------
        surface_channel_results : figure
            A figure summarizing the results of seeking the surface channel with LFP data.
        ----------
        """

        nchan = int(kwargs['nchan'] if 'nchan' in kwargs.keys() and (type(kwargs['nchan']) == int or type(kwargs['nchan']) == float) else 385)
        lfp_sampling_frequency = int(kwargs['lfp_sampling_frequency'] if 'lfp_sampling_frequency' in kwargs.keys() and (type(kwargs['lfp_sampling_frequency']) == int or type(kwargs['lfp_sampling_frequency']) == float) else 2.5e3)
        lfp_gain_setting = int(kwargs['lfp_gain_setting'] if 'lfp_gain_setting' in kwargs.keys() and (type(kwargs['lfp_gain_setting']) == int or type(kwargs['lfp_gain_setting']) == float) else 250)
        smoothing_width = kwargs['smoothing_width'] if 'smoothing_width' in kwargs.keys() and (type(kwargs['smoothing_width']) == int or type(kwargs['smoothing_width']) == float) else 5
        power_threshold = kwargs['power_threshold'] if 'power_threshold' in kwargs.keys() and (type(kwargs['power_threshold']) == int or type(kwargs['power_threshold']) == float) else 2.5
        diff_threshold = kwargs['diff_threshold'] if 'diff_threshold' in kwargs.keys() and (type(kwargs['diff_threshold']) == int or type(kwargs['diff_threshold']) == float) else -0.06
        freq_range = kwargs['freq_range'] if 'freq_range' in kwargs.keys() and type(kwargs['freq_range']) == list else [0, 10]
        channel_range = kwargs['channel_range'] if 'channel_range' in kwargs.keys() and type(kwargs['channel_range']) == list else False
        nfft = kwargs['nfft'] if 'nfft' in kwargs.keys() and type(kwargs['nfft']) == int else 4096
        n_passes = kwargs['n_passes'] if 'n_passes' in kwargs.keys() and type(kwargs['n_passes']) == int else 5
        skip_s_per_pass = kwargs['skip_s_per_pass'] if 'skip_s_per_pass' in kwargs.keys() and type(kwargs['skip_s_per_pass']) == int else 5
        max_freq = kwargs['max_freq'] if 'max_freq' in kwargs.keys() and type(kwargs['max_freq']) == int else 150
        reference_channel = kwargs['reference_channel'] if 'reference_channel' in kwargs.keys() else False
        to_plot = kwargs['to_plot'] if 'to_plot' in kwargs.keys() else 1
        colormap = kwargs['colormap'] if 'colormap' in kwargs.keys() and type(kwargs['colormap']) == str else 'afmhot'
        save_plot = kwargs['save_plot'] if 'save_plot' in kwargs.keys() else 0
        fig_format = kwargs['fig_format'] if 'fig_format' in kwargs.keys() and type(kwargs['fig_format']) == str else 'png'

        # load LFP data
        for file in os.listdir(self.lfp_dir):
            if 'lf' in file and 'bin' in file:

                print('Working on file: {}{}{}.'.format(self.lfp_dir, os.sep, file))

                # load file into memory
                lfp_recording = np.memmap('{}{}{}'.format(self.lfp_dir, os.sep, file), mode='r', dtype=np.int16, order='C')

                # find the total number of LFP samples
                lfp_samples = lfp_recording.shape[0] // nchan

                # reshape array, but because channels are columns, the order stays C
                lfp_data = lfp_recording.reshape((lfp_samples, nchan), order='C')

                # depending on the gain settings, convert data to voltage
                if lfp_gain_setting == 250:
                    lfp_data = lfp_data * 4.69
                elif lfp_gain_setting == 125:
                    lfp_data = lfp_data * 9.38
                else:
                    print('Unrecognized gain setting: {}, try again.'.format(lfp_gain_setting))
                    sys.exit()
                break
        else:
            print('Could not find LFP file in directory: {}, try again.'.format(self.lfp_dir))
            sys.exit()

        # initialize array based on number of passes
        candidates = np.zeros((n_passes,))

        for p in range(n_passes):

            # select one second of data
            start_part = int(lfp_sampling_frequency*skip_s_per_pass*p)
            end_part = start_part + int(lfp_sampling_frequency)

            channels = np.arange(nchan-1).astype('int')

            chunk = np.copy(lfp_data[start_part:end_part, channels])

            # median subtract every channel
            for ch in np.arange(nchan-1):
                chunk[:, ch] = chunk[:, ch] - np.median(chunk[:, ch])

            # median subtract saline channels together (if present)
            if type(channel_range) == list:
                for ch in np.arange(nchan-1):
                    chunk[:, ch] = chunk[:, ch] - np.median(chunk[:, channel_range[0]:channel_range[1]], 1)

            # Estimate power spectral density using Welch’s method.
            # Welch’s method computes an estimate of the power spectral density by dividing the data into overlapping segments,
            # computing a modified periodogram for each segment and averaging the periodograms
            power = np.zeros((int(nfft/2+1), nchan-1))
            for ch in np.arange(nchan-1):
                sample_frequencies, px_den = welch(chunk[:, ch], fs=lfp_sampling_frequency, nfft=nfft)
                power[:, ch] = px_den

            # find frequencies for plotting purposes
            in_range = np.where((sample_frequencies >= 0) * (sample_frequencies <= max_freq))[0]
            freqs_to_plot = sample_frequencies[in_range]

            # average and tak log of powers in the specified frequency band
            in_range_freqs = np.where((sample_frequencies >= freq_range[0]) * (sample_frequencies <= freq_range[1]))[0]
            values = np.log10(np.mean(power[in_range_freqs, :], 0))

            # mask reference channel (if present)
            if type(reference_channel) != bool:
                values[reference_channel] = values[reference_channel-1]

            # smooth results
            values = gaussian_filter1d(values, smoothing_width)

            # select possible surface channel
            surface_channels = np.where((np.diff(values) < diff_threshold) * (values[:-1] < power_threshold) )[0]

            if surface_channels.shape[0] > 0:
                candidates[p] = np.max(surface_channels)
            else:
                candidates[p] = nchan-1

        # go through passes and find best candidate for surface channel
        surface_channel = int(round(np.median(candidates)))

        print('The likeliest candidate for the surface channel is {} (with channel count started from 0).'.format(surface_channel))

        if to_plot:
            fig, ax = plt.subplots(2, 2, figsize=(10, 6), dpi=300)
            plt.subplots_adjust(hspace=.45, wspace=.35)
            ax1 = plt.subplot(2, 2, 1)
            im1 = ax1.imshow((chunk).T[::-1], aspect='auto', cmap=colormap)
            cbar1 = fig.colorbar(im1, label='LFP signal ($\mu$V)')
            cbar1.ax.tick_params(size=0)
            ax1.set_title('Median subtracted LFP data')
            ax1.set_xlabel('Time (s)')
            ax1_xticks = np.arange(0, (end_part - start_part + 1), 500)
            ax1.set_xticks(ax1_xticks)
            ax1.set_xticklabels(['{}'.format(xt / (end_part - start_part)) for xt in ax1_xticks])
            ax1.set_ylabel('Channel ID')
            ax1.set_yticks(np.arange(0, (nchan - 1), 50))
            ax1.set_yticklabels(['{}'.format(xt) for xt in np.arange((nchan - 1), 0, -50)])
            ax1.tick_params(axis='both', which='both', length=0)

            ax2 = plt.subplot(2, 2, 3)
            im2 = ax2.imshow(np.log10(power[in_range, :]).T[::-1], aspect='auto', cmap=colormap)
            cbar2 = fig.colorbar(im2, label='$log_{10} power$')
            cbar2.ax.tick_params(size=0)
            ax2.set_title('Power spectrum')
            ax2.set_xlabel('Frequency (Hz)')
            freq_range = np.concatenate((freqs_to_plot[::41], freqs_to_plot[-1::freqs_to_plot.shape[0]]))
            ax2_xticks = np.isin(freqs_to_plot, freq_range)
            ax2.set_xticks(in_range[ax2_xticks])
            ax2.set_xticklabels(['{}'.format(int(round(xt))) for xt in freq_range])
            ax2.set_ylabel('Channel ID')
            ax2.set_yticks(np.arange(0, (nchan - 1), 50))
            ax2.set_yticklabels(['{}'.format(xt) for xt in np.arange((nchan - 1), 0, -50)])
            ax2.tick_params(axis='both', which='both', length=0)

            ax3 = plt.subplot(2, 2, 2)
            ax3.plot(values, color='#008B45')
            ax3.axhline(power_threshold, ls='--', color='#000000', label='power threshold')
            ax3.axvline(surface_channel, ls='--', color='#FF6347', label='surface channel')
            ax3.set_title('Power measure')
            ax3.set_xlabel('Channel ID')
            ax3_xticks = np.arange(0, (nchan - 1), 50)
            ax3.set_xticks(ax3_xticks)
            ax3.set_xticklabels(['{}'.format(xt) for xt in ax3_xticks])
            ax3.set_ylabel('Log mean power')
            ax3.tick_params(axis='both', which='both', length=0)
            ax3.legend(loc='best', fontsize='x-small')

            ax4 = plt.subplot(2, 2, 4)
            ax4.plot(np.diff(values), color='#008B45')
            ax4.axhline(diff_threshold, ls='--', color='#000000', label='difference threshold')
            ax4.axvline(surface_channel, ls='--', color='#FF6347', label='surface channel')
            ax4.set_title('Difference measure')
            ax4.set_xlabel('Channel comparison')
            ax4_xticks = np.arange(0, (nchan - 1), 50)
            ax4.set_xticks(ax4_xticks)
            ax4.set_xticklabels(['{}'.format(xt) for xt in ax4_xticks])
            ax4.set_ylabel('Log mean power differences')
            ax4.tick_params(axis='both', which='both', length=0)
            ax4.legend(loc='best', fontsize='x-small')
            fig.text(x=.4, y=.05, s='surface channel: {:d}'.format(surface_channel))
            if save_plot:
                fig.savefig('{}{}surface_channel_results.{}'.format(self.lfp_dir, os.sep, fig_format))
            plt.show()
