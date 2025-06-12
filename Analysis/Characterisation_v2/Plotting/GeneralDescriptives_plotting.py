import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'
from scipy.signal import medfilt
import matplotlib.ticker as ticker
import os
import pickle

from Helpers.Config_23 import *

from Analysis.Characterisation_v2 import General_utils as gu
from Analysis.Characterisation_v2 import Plotting_utils as pu

def plot_literature_parallels(feature_data, stride, phases, savedir, fs=7):
    features = ['stride_length|speed_correct:True', 'cadence', 'walking_speed|bodypart:Tail1, speed_correct:True',
                'bos_stancestart|ref_or_contr:ref, y_or_euc:y', 'bos_stancestart|ref_or_contr:contr, y_or_euc:y',
                'back_skew|step_phase:1, all_vals:False, full_stride:False, buffer_size:0',
                'signed_angle|Back1Back12_side_zref_swing_peak']
    for feature in features:
        # Plot a TS of feature and also phase difference
        single_feat = feature_data.loc(axis=0)[stride].loc(axis=1)[feature]
        # Get the mask for each phase
        mask_p1, mask_p2 = gu.get_mask_p1_p2(single_feat, 'APA1', 'APA2')
        featsp1 = single_feat.loc(axis=0)[mask_p1]
        featsp2 = single_feat.loc(axis=0)[mask_p2]
        # make mice a column
        featsp1 = featsp1.unstack(level=0)
        featsp2 = featsp2.unstack(level=0)

        feat_name = short_names.get(feature, feature)

        # Plot the feature
        # fig, axs = plt.subplots(1, 2, figsize=(5, 2))
        fig, axs = plt.subplots(
            1, 2,
            figsize=(3, 2),
            gridspec_kw={'width_ratios': [2, 1], 'wspace': 0.2}
        )

        apa1_color = pu.get_color_phase('APA1')
        apa2_color = pu.get_color_phase('APA2')
        wash1_color = pu.get_color_phase('Wash1')
        wash2_color = pu.get_color_phase('Wash2')

        boxy = 1
        height = 0.02
        patch1 = axs[0].axvspan(xmin=10, xmax=60, ymin=boxy, ymax=boxy + height, color=apa1_color, lw=0)
        patch2 = axs[0].axvspan(xmin=60, xmax=110, ymin=boxy, ymax=boxy + height, color=apa2_color, lw=0)
        patch3 = axs[0].axvspan(xmin=110, xmax=135, ymin=boxy, ymax=boxy + height, color=wash1_color, lw=0)
        patch4 = axs[0].axvspan(xmin=135, xmax=160, ymin=boxy, ymax=boxy + height, color=wash2_color, lw=0)
        patch1.set_clip_on(False)
        patch2.set_clip_on(False)
        patch3.set_clip_on(False)
        patch4.set_clip_on(False)

        mice = single_feat.index.get_level_values(level='MouseID').unique().tolist()
        common_x = np.arange(0, 160)

        # plot time series
        mice_data = np.zeros((len(mice), len(common_x)))
        for midx, m in enumerate(mice):
            mouse_feat = single_feat.loc(axis=0)[m]
            mouse_feat_interp = np.interp(common_x+1, mouse_feat.index.get_level_values(level='Run'), mouse_feat.values)
            mouse_feat_smooth = medfilt(mouse_feat_interp, kernel_size=15)
            #axs[0].plot(common_x, mouse_feat_smooth, alpha=0.2, color='grey')
            mice_data[midx, :] = mouse_feat_smooth
        mice_mean = np.mean(mice_data, axis=0)
        mice_sem = np.std(mice_data, axis=0) / np.sqrt(len(mice))
        axs[0].plot(common_x+1, mice_mean, color='black', label='Mean', linewidth=1)
        axs[0].fill_between(common_x, mice_mean - mice_sem, mice_mean + mice_sem, color='black', alpha=0.2)

        axs[0].axvline(x=10, color='k', linestyle='--', alpha=0.2)
        axs[0].axvline(x=60, color='k', linestyle='--', alpha=0.2)
        axs[0].axvline(x=110, color='k', linestyle='--', alpha=0.2)
        axs[0].axvline(x=135, color='k', linestyle='--', alpha=0.2)

        axs[0].grid(False)
        axs[0].spines['top'].set_visible(False)
        axs[0].spines['right'].set_visible(False)
        axs[0].set_ylim(-1.2, 1.2)
        axs[0].set_yticks([-1, 0, 1])
        axs[0].tick_params(axis='x', which='major', bottom=True, top=False, length=4, width=1)
        axs[0].tick_params(axis='x', which='minor', bottom=True, top=False, length=2, width=1)
        axs[0].set_yticklabels([-1, 0, 1], fontsize=fs)
        axs[0].set_ylabel(f"{feat_name} (z-scored)", fontsize=fs)
        axs[0].set_xlabel('Run', fontsize=fs)
        axs[0].set_xlim(0, 160)
        axs[0].set_xticks([0, 50, 100, 150])
        axs[0].set_xticklabels([0, 50, 100, 150], fontsize=fs)
        axs[0].xaxis.set_minor_locator(ticker.MultipleLocator(10))

        axs[0].yaxis.set_minor_locator(ticker.MultipleLocator(0.1))  # where the ticks go
        axs[0].tick_params(axis='y', which='minor',
                           left=True,  # ← use left / right for the y‑axis
                           right=False,
                           length=2, width=1)
        axs[0].tick_params(axis='y', which='major',
                           left=True, right=False,
                           length=4, width=1)

        # plt differences between phase
        p1_means = np.zeros((len(mice)))
        p2_means = np.zeros((len(mice)))
        for midx, m in enumerate(mice):
            m_p1 = featsp1.loc(axis=1)[m]
            m_p2 = featsp2.loc(axis=1)[m]
            p1_mean = m_p1.mean()
            p2_mean = m_p2.mean()
            p1_means[midx] = p1_mean
            p2_means[midx] = p2_mean
            axs[1].plot([0, 1], [p1_mean, p2_mean], 'o-', alpha=0.3, color='grey', markersize=3, zorder=10)

        # plot boxplots
        # Get the phase colours and darker versions for the median and whiskers
        p1_color = pu.get_color_phase(phases[0])
        p2_color = pu.get_color_phase(phases[1])
        dark_color_p1 = pu.darken_color(p1_color, 0.7)
        dark_color_p2 = pu.darken_color(p2_color, 0.7)

        # Boxplot properties for phases
        boxprops_p1 = dict(facecolor=p1_color, color=p1_color)
        boxprops_p2 = dict(facecolor=p2_color, color=p2_color)
        medianprops_p1 = dict(color=dark_color_p1, linewidth=2)
        whiskerprops_p1 = dict(color=dark_color_p1, linewidth=1.5, linestyle='-')
        medianprops_p2 = dict(color=dark_color_p2, linewidth=2)
        whiskerprops_p2 = dict(color=dark_color_p2, linewidth=1.5, linestyle='-')

        x = np.array([0.5])
        width = 0.35
        bar_multiple = 0.6
        positions_p1 = np.array([0])#x - width / 2
        positions_p2 = np.array([1])#x + width / 2

        axs[1].boxplot(p1_means, positions=positions_p1, widths=width*bar_multiple,
                       patch_artist=True, boxprops=boxprops_p1,
                       medianprops=medianprops_p1, whiskerprops=whiskerprops_p1, showcaps=False, showfliers=False)
        axs[1].boxplot(p2_means, positions=positions_p2, widths=width*bar_multiple,
                          patch_artist=True, boxprops=boxprops_p2,
                          medianprops=medianprops_p2, whiskerprops=whiskerprops_p2, showcaps=False, showfliers=False)

        axs[1].set_xticks([0, 1])
        axs[1].set_xticklabels(
            [r'APA$_{\mathrm{end}}$', r'Wash$_{\mathrm{end}}$'],
            fontsize=fs
        )
        axs[1].xaxis.set_minor_locator(ticker.MultipleLocator(0.1))  # where the ticks go
        # axs[1].tick_params(axis='x', which='minor',
        #                     bottom=True, top=False,
        #                     length=2, width=1)
        axs[1].tick_params(axis='x', which='major',
                            bottom=True, top=False,
                            length=4, width=1)
        axs[1].set_xlim(-0.5, 1.5)
        axs[1].set_ylim(-1, 1)
        axs[1].set_yticks([-1, 0, 1])
        axs[1].set_yticklabels([-1, 0, 1], fontsize=fs)
        axs[1].yaxis.set_minor_locator(ticker.MultipleLocator(0.1))  # where the ticks go
        axs[1].tick_params(axis='y', which='minor',
                            left=True,  # ← use left / right for the y‑axis
                            right=False,
                            length=2, width=1)
        axs[1].tick_params(axis='y', which='major',
                            left=True, right=False,
                            length=4, width=1)

        axs[1].grid(False)
        axs[1].spines['top'].set_visible(False)
        axs[1].spines['right'].set_visible(False)

        fig.subplots_adjust(left=0.15, right=0.99, top=0.9, bottom=0.2)

        fig.savefig(os.path.join(savedir, f"{feat_name}_{stride}_{phases[0]}_{phases[1]}.png"), dpi=300)
        fig.savefig(os.path.join(savedir, f"{feat_name}_{stride}_{phases[0]}_{phases[1]}.svg"), dpi=300)
        plt.close(fig)




def plot_angles(feature_data_notscaled, phases, stride, savedir):
    features_to_plot = ['signed_angle|ToeKnuckle_ipsi_side_zref_swing_mean',
                        'signed_angle|ToeKnuckle_ipsi_side_zref_swing_peak',
                        'signed_angle|ToeKnuckle_contra_side_zref_swing_mean',
                        'signed_angle|ToeKnuckle_contra_side_zref_swing_peak',
                        'signed_angle|ToeAnkle_ipsi_side_zref_swing_mean',
                        'signed_angle|ToeAnkle_ipsi_side_zref_swing_peak',
                        'signed_angle|ToeAnkle_contra_side_zref_swing_mean',
                        'signed_angle|ToeAnkle_contra_side_zref_swing_peak',
                        'signed_angle|Back1Back12_side_zref_swing_mean',
                        'signed_angle|Back1Back12_side_zref_swing_peak',
                        'signed_angle|Tail1Tail12_side_zref_swing_mean',
                        'signed_angle|Tail1Tail12_side_zref_swing_peak',
                        'signed_angle|NoseBack1_side_zref_swing_mean',
                        'signed_angle|NoseBack1_side_zref_swing_peak',
                        'signed_angle|Back1Back12_overhead_xref_swing_mean',
                        'signed_angle|Back1Back12_overhead_xref_swing_peak',
                        'signed_angle|Tail1Tail12_overhead_xref_swing_mean',
                        'signed_angle|Tail1Tail12_overhead_xref_swing_peak',
                        'signed_angle|NoseBack1_overhead_xref_swing_mean',
                        'signed_angle|NoseBack1_overhead_xref_swing_peak'
                        ]
    for feature in features_to_plot:
        data_notscaled = feature_data_notscaled.loc(axis=1)[feature]
        # Get the mask for each phase
        mask_p1, mask_p2 = gu.get_mask_p1_p2(data_notscaled, phases[0], phases[1])
        plot_angle_polar(angle_p1=data_notscaled[mask_p1],angle_p2=data_notscaled[mask_p2],
                         p1=phases[0], p2=phases[1], stride=stride, feature=feature, savedir=savedir)

def plot_angle_polar(angle_p1, angle_p2 ,p1, p2, stride, feature, savedir):
    short_feat = short_names.get(feature, feature)
    # Group by run (assuming the run is on level=1) to compute the average across all observations per run
    angle_p1_avg = angle_p1.groupby(level=1).mean()
    angle_p2_avg = angle_p2.groupby(level=1).mean()
    # Convert from degrees to radians as polar histograms require radians
    theta_p1 = np.deg2rad(angle_p1_avg.values)
    theta_p2 = np.deg2rad(angle_p2_avg.values)
    # Define the bins for the polar histogram
    # Here we create 20 equal bins spanning from 0 to 2*pi
    num_bins = 90
    bins = np.linspace(0, 2 * np.pi, num_bins + 1)
    # Compute histograms: counts of run averages falling within each bin
    hist_p1, _ = np.histogram(theta_p1, bins=bins)
    hist_p2, _ = np.histogram(theta_p2, bins=bins)
    # The width of each bin, needed for the bar plot
    width = bins[1] - bins[0]
    # Create a figure with two polar subplots side by side
    fig, ax = plt.subplots(1, 1, subplot_kw={'projection': 'polar'}, figsize=(5, 5))
    # Plot Phase 1: each bar's length corresponds to the number of runs in that angular bin
    p1_col = pu.get_color_phase(p1)
    p2_col = pu.get_color_phase(p2)
    ax.bar(bins[:-1], hist_p1, width=width, bottom=0.0, align='edge', linewidth=0, color=p1_col, alpha=0.6, label=p1)
    ax.set_title(short_feat)
    # Plot Phase 2
    ax.bar(bins[:-1], hist_p2, width=width, bottom=0.0, align='edge', linewidth=0, color=p2_col, alpha=0.6, label=p2)
    if 'yaw' in short_feat:
        ax.set_theta_zero_location("W")
    elif 'pitch' in short_feat:
        ax.set_theta_zero_location("S")
    ax.set_theta_direction(1)
    # ax.set_thetamin(180)
    # ax.set_thetamax(0)
    plt.legend()
    save_path = os.path.join(savedir, f"polar_histogram_{short_feat}_{p1}_{p2}_{stride}")
    plt.savefig(f"{save_path}.png", dpi=300)
    plt.savefig(f"{save_path}.svg", dpi=300)
    plt.close()

# Import data
with open(r"H:\\Characterisation\\LH_res_-3-2-1_APA2Wash2-PCStot=60-PCSuse=12\\preprocessed_data_APAChar_LowHigh.pkl",
          'rb') as f:
    data_LH = pickle.load(f)