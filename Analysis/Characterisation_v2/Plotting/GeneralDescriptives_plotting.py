import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'
from scipy.signal import medfilt
import matplotlib.ticker as ticker
import os
import pickle
from matplotlib.patches import Patch
from scipy.interpolate import interp1d
import re

from Helpers.Config_23 import *

from Analysis.Characterisation_v2 import General_utils as gu
from Analysis.Characterisation_v2 import Plotting_utils as pu


def extract_bodypart_labels(feature_name, bodypart_names):
    # Sort by length descending so Back12 is checked before Back1, etc.
    sorted_names = sorted(bodypart_names, key=len, reverse=True)
    matches = []
    used_ranges = []

    idx = 0
    while len(matches) < 2 and idx < len(feature_name):
        found = False
        for bp in sorted_names:
            if feature_name.startswith(bp, idx):
                matches.append(bp)
                idx += len(bp)
                found = True
                break
        if not found:
            idx += 1  # move forward in string if no match at current position

    # If two matches found, return in order of appearance
    if len(matches) >= 2:
        return matches[0], matches[1]
    elif len(matches) == 1:
        return matches[0], ''
    else:
        return '', ''


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
    feature_name = angle_p1.name
    bodypart_names = ['Toe', 'Knuckle', 'Ankle', 'Back1', 'Back12', 'Tail1', 'Tail12', 'Nose']
    short_feat = short_names.get(feature, feature)

    angle_p1.index = angle_p1.index.set_names(['Stride', 'MouseID', 'FrameIdx'])
    angle_p2.index = angle_p2.index.set_names(['Stride', 'MouseID', 'FrameIdx'])

    # filter by stride
    angle_p1 = angle_p1.loc(axis=0)[stride]
    angle_p2 = angle_p2.loc(axis=0)[stride]

    if 'ToeKnuckle' in feature_name or 'ToeAnkle' in feature_name:
        # Flip the angles as they were accidentally calculated Knuckle-Toe and Ankle-Toe
        angle_p1 = -angle_p1
        angle_p2 = -angle_p2
    if 'overhead_xref' in feature_name:
        # so 0 degrees is facing forward, not backward
        angle_p1 = -angle_p1
        angle_p2 = -angle_p2

    # Group by run (assuming the run is on level=1) to compute the average across all observations per run
    angle_p1_avg = angle_p1.groupby(level='MouseID').mean()
    angle_p2_avg = angle_p2.groupby(level='MouseID').mean()
    # Convert from degrees to radians as polar histograms require radians
    theta_p1 = np.deg2rad(angle_p1_avg.values)
    theta_p2 = np.deg2rad(angle_p2_avg.values)
    # Define the bins for the polar histogram
    # Here we create 20 equal bins spanning from 0 to 2*pi
    num_bins = 180
    bins = np.linspace(-np.pi, np.pi, num_bins + 1)
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

    ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    ax.set_xticklabels(['–180°', '–90°', '0°', '90°', '180°'])

    # Extract labels automatically
    bp1, bp2 = extract_bodypart_labels(feature_name, bodypart_names)

    # Center label
    ax.text(0, 0, bp2, ha='center', va='center', fontsize=12, fontweight='bold', color='dimgray')

    # Outer label at 0 radians
    r = ax.get_rmax() * 1.1
    ax.text(0, r, bp1, ha='center', va='bottom', fontsize=12, color='dimgray')

    if 'yaw' in short_feat:
        ax.set_theta_zero_location("E")
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

def plot_limb_positions(raw_data, phases, savedir, fs=7):
    forelimb_parts = ['ForepawToe', 'ForepawKnuckle', 'ForepawAnkle', 'ForepawKnee']
    y_midline = structural_stuff['belt_width'] / 2

    for phase in phases:
        # Get data from last stride before transition
        for midx, mouse in enumerate(raw_data.keys()):
            mouse_data = raw_data[mouse]
            # Drop the 'Day' index level
            mouse_data = mouse_data.droplevel('Day', axis=0)
            mouse_data = mouse_data.loc(axis=0)[expstuff['condition_exp_runs']['APAChar']['Extended'][phase]]
            for run in mouse_data.index.get_level_values('Run').unique():
                run_data = mouse_data.loc(axis=0)[run]
                transition_idx = run_data.index.get_level_values(level='FrameIdx')[run_data.index.get_level_values('RunStage') == 'Transition'][0]
                transition_paw = run_data.loc(axis=1)['initiating_limb'].loc(axis=0)['Transition', transition_idx]

                # Get the last stride before transition
                stance_periods_mask = run_data.loc(axis=0)['RunStart'].loc(axis=1)[transition_paw,'SwSt_discrete'] == locostuff['swst_vals_2025']['st']
                stance_periods_idxs = run_data.loc(axis=0)['RunStart'].index.get_level_values(level='FrameIdx')[stance_periods_mask]
                last_stance_idx = stance_periods_idxs[-1]
                stride_data = run_data.loc(axis=0)['RunStart',np.arange(last_stance_idx,transition_idx)]

                fig, ax = plt.subplots(figsize=(5, 2.5))
                column_multi = pd.MultiIndex.from_product([forelimb_parts, ['x', 'y', 'z']], names=['BodyPart', 'Coord'])
                run_coords = pd.DataFrame(index=stride_data.index.get_level_values(level='FrameIdx'), columns=column_multi)
                for fidx, frame in enumerate(stride_data.index.get_level_values('FrameIdx')):
                    frame_data = stride_data.loc(axis=0)['RunStart',frame]
                    transition_paw_side = transition_paw.split('Forepaw')[1]  # 'R' or 'L'

                    transition_paw_limbparts = [l+transition_paw_side for l in forelimb_parts]
                    x = frame_data.loc[transition_paw_limbparts, 'x'].values
                    y = frame_data.loc[transition_paw_limbparts, 'y'].values
                    z = frame_data.loc[transition_paw_limbparts, 'z'].values

                    if transition_paw_side == 'L':
                        # Need to flip the y coords to mirror the left side to the right side
                        mirrored_y = 2 * y_midline - y
                        y = mirrored_y

                    # Plot the limb positions
                    ax.plot(x, z, marker='o', markersize=2, color='grey', alpha=0.5, linewidth=0.5)
                    # Store the coordinates for the current frame
                    col_x = [(part, 'x') for part in forelimb_parts]
                    col_y = [(part, 'y') for part in forelimb_parts]
                    col_z = [(part, 'z') for part in forelimb_parts]

                    run_coords.loc[frame, col_x] = x
                    run_coords.loc[frame, col_y] = y
                    run_coords.loc[frame, col_z] = z

                run_coords_mice_mean = run_coords.mean(axis=0)
                # Plot the mean limb positions
                ax.plot(run_coords_mice_mean.loc(axis=0)[forelimb_parts, 'x'], run_coords_mice_mean.loc(axis=0)[forelimb_parts, 'z'],
                        marker='o', markersize=3, color='black', linewidth=1, label='Mean')


def plot_limb_positions_average(raw_data, mouse_runs, phases, savedir, fs=7, n_interp=100):
    forelimb_parts = ['ForepawToe', 'ForepawKnuckle', 'ForepawAnkle', 'ForepawKnee']
    mice = ['1035243', '1035244', '1035245', '1035246', '1035250', '1035297', '1035299', '1035301']
    coords = ['x', 'y', 'z']
    y_midline = structural_stuff['belt_width'] / 2

    all_arr = {}
    all_stride_lengths = []
    for phase in phases:
        phase_runs = expstuff['condition_exp_runs']['APAChar']['Extended'][phase]
        n_phase_runs = len(phase_runs)
        n_mice = len(mice)
        n_bps = len(forelimb_parts)
        arr = np.full((n_mice, n_phase_runs, n_bps, 3, n_interp), np.nan)

        for m, mouse in enumerate(mice):
            mouse_data = raw_data[mouse].droplevel('Day', axis=0)
            for r, run in enumerate(phase_runs):
                if run not in mouse_runs[mouse]:
                    print(f"Run {run} not in mouse {mouse} runs, skipping.")
                    continue

                run_data = mouse_data.loc(axis=0)[run]
                transition_idx = run_data.index.get_level_values(level='FrameIdx')[run_data.index.get_level_values('RunStage') == 'Transition'][0]
                transition_paw = run_data.loc(axis=1)['initiating_limb'].loc(axis=0)['Transition', transition_idx]
                stance_periods_mask = run_data.loc(axis=0)['RunStart'].loc(axis=1)[transition_paw, 'SwSt_discrete'] == locostuff['swst_vals_2025']['st']
                stance_periods_idxs = run_data.loc(axis=0)['RunStart'].index.get_level_values(level='FrameIdx')[stance_periods_mask]
                last_stance_idx = stance_periods_idxs[-1]
                stride_data = run_data.loc(axis=0)['RunStart', np.arange(last_stance_idx, transition_idx)]

                bp_side = transition_paw.split('Forepaw')[1]  # 'L' or 'R'

                # 1. Reference for x: get from initiating paw at stride start and end
                ref_bp = 'ForepawToe' + bp_side  # e.g., 'ForepawToeR'
                x_ref_all = stride_data.loc(axis=1)[ref_bp, 'x'].values
                x_start = x_ref_all[0]
                x_end = x_ref_all[-1]

                # 2. Stack all joints to get overall mean y/z for run (use a list or array)
                y_all_joints = []
                z_all_joints = []
                for bp in forelimb_parts:
                    bp_name = bp + bp_side
                    try:
                        y = stride_data.loc(axis=1)[bp_name, 'y'].values
                        z = stride_data.loc(axis=1)[bp_name, 'z'].values
                        y_all_joints.append(y)
                        z_all_joints.append(z)
                    except KeyError:
                        continue
                y_all_joints = np.concatenate(y_all_joints)
                z_all_joints = np.concatenate(z_all_joints)
                y_center = np.mean(y_all_joints)
                z_center = np.mean(z_all_joints)

                for b, bp in enumerate(forelimb_parts):
                    bp_name = bp + bp_side
                    x = stride_data.loc(axis=1)[bp_name, 'x'].values
                    y = stride_data.loc(axis=1)[bp_name, 'y'].values
                    z = stride_data.loc(axis=1)[bp_name, 'z'].values

                    if np.isnan(x).any() or np.isnan(y).any() or np.isnan(z).any():
                        continue
                    if bp_side == 'L':
                        y = 2 * y_midline - y
                    n_pts = len(x)

                    # --- NORMALISE ---
                    # x: relative to stride start/end of reference toe/ankle
                    x_norm = 100 * (x - x_start) / (x_end - x_start)
                    # y/z: center by whole run mean
                    y_norm = y - y_center
                    z_norm = z - z_center

                    interp_x = interp1d(np.linspace(0, 1, n_pts), x_norm, kind='linear')(np.linspace(0, 1, n_interp))
                    interp_y = interp1d(np.linspace(0, 1, n_pts), y_norm, kind='linear')(np.linspace(0, 1, n_interp))
                    interp_z = interp1d(np.linspace(0, 1, n_pts), z_norm, kind='linear')(np.linspace(0, 1, n_interp))
                    arr[m, r, b, 0, :] = interp_x
                    arr[m, r, b, 1, :] = interp_y
                    arr[m, r, b, 2, :] = interp_z


        # --- NEW: Stick-figure x-z trajectory plot for this phase ---
        # 1. Compute the average x and z for each joint at each time point
        mean_x = np.nanmean(arr[:, :, :, 0, :], axis=(0, 1))  # shape: [n_bps, n_interp]
        mean_y = np.nanmean(arr[:, :, :, 1, :], axis=(0, 1))  # shape: [n_bps, n_interp]
        mean_z = np.nanmean(arr[:, :, :, 2, :], axis=(0, 1))  # shape: [n_bps, n_interp]

        # 2. Stick-figure sequence plot (trajectory through the stride)
        fig, ax = plt.subplots(figsize=(6, 2))
        # Plot stick figures for each timepoint (optional: stride through in steps for less clutter)
        for t in range(n_interp):
            xs = mean_x[:, t]
            zs = mean_z[:, t]
            ax.plot(xs, zs, marker='.', color='k', alpha=0.4, linewidth=0.5, markersize=1, zorder=1)
        ax.set_title(phase, fontsize=fs)
        ax.set_xlabel('x (normalised)', fontsize=fs)
        ax.set_ylabel('z (centred)', fontsize=fs)
        ax.tick_params(axis='both', which='major', labelsize=fs)
        ax.set_ylim(-10, 10)
        ax.set_yticks(np.arange(-10, 11, 5))
        ax.set_yticklabels(np.arange(-10, 11, 5), fontsize=fs)
        ax.set_xlim(-50, 100)
        ax.set_xticks(np.arange(-50, 101, 50))
        ax.set_xticklabels(np.arange(-50, 101, 50), fontsize=fs)
        ax.set_aspect('equal')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        save_path = os.path.join(savedir, f"stick_trajectory_{phase}")
        plt.savefig(f"{save_path}.png", dpi=400)
        plt.savefig(f"{save_path}.svg", dpi=400)
        plt.close(fig)

        # plot 3d version
        # 3D plot of the stick figure trajectory
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection='3d')
        for t in range(n_interp):
            xs = mean_x[:, t]
            ys = mean_y[:, t]
            zs = mean_z[:, t]
            ax.plot(xs, ys, zs, marker='.', color='k', alpha=0.4, linewidth=0.5, markersize=1, zorder=1)
        ax.set_xlim(np.nanmin(mean_x), np.nanmax(mean_x))
        ax.set_ylim(np.nanmin(mean_y), np.nanmax(mean_y))
        ax.set_zlim(np.nanmin(mean_z), np.nanmax(mean_z))

        ax.set_title(phase, fontsize=fs)
        ax.set_xlabel('x (normalised)', fontsize=fs)
        ax.set_ylabel('y (centred)', fontsize=fs)
        ax.set_zlabel('z (centred)', fontsize=fs)
        ax.tick_params(axis='both', which='major', labelsize=fs)
        # Calculate the ranges for each axis
        x_range = np.nanmax(mean_x) - np.nanmin(mean_x)
        y_range = np.nanmax(mean_y) - np.nanmin(mean_y)
        z_range = np.nanmax(mean_z) - np.nanmin(mean_z)
        ax.set_box_aspect([x_range, y_range, z_range])  # Matches real proportions!
        #        plt.tight_layout()
        plt.savefig(f"{savedir}/stick_trajectory_3d_{phase}.png", dpi=400)
        plt.close(fig)

        # 3. Mean stick-figure (average posture across stride)
        # Recenter x to the toe at each timepoint before averaging
        arr_toe_x = arr[:, :, 0, 0, :]  # [mouse, run, time] -- toe's x
        # Subtract the toe x at each timepoint from all bodyparts for each run/mouse
        arr_x_centered = arr[:, :, :, 0, :] - arr_toe_x[:, :, np.newaxis, :]
        # Now avg_x gives you mean shape with toe at x=0 for each timepoint
        mean_x_centered = np.nanmean(arr_x_centered, axis=(0, 1))  # shape: [n_bps, n_interp]
        avg_x = np.nanmean(mean_x_centered, axis=1)
        avg_z = np.nanmean(mean_z, axis=1)
        std_x = np.nanstd(mean_x_centered, axis=1)
        std_z = np.nanstd(mean_z, axis=1)


        fig, ax = plt.subplots(figsize=(2.5, 2))
        ax.plot(avg_x, avg_z, marker='o', color='black', linewidth=1, markersize=2, zorder=2)
        # Plot error bars for std deviation
        eb = ax.errorbar(avg_x, avg_z, xerr=std_x, yerr=std_z, fmt='none', ecolor='grey', elinewidth=0.5, capsize=2, zorder=1)
        for bar in eb[2]:
            bar.set_linestyle('--')

        ax.set_title(phase, fontsize=fs)
        ax.set_xlabel('x (normalised)', fontsize=fs)
        ax.set_ylabel('z (centred)', fontsize=fs)
        ax.set_ylim(-8,8)
        ax.set_yticks(np.arange(-8, 9, 8))
        ax.set_yticklabels(np.arange(-8, 9, 8), fontsize=fs)
        ax.set_xlim(-40, 10)
        ax.set_xticks(np.arange(-40, 11, 10))
        ax.set_xticklabels(np.arange(-40, 11, 10), fontsize=fs)
        ax.set_aspect('equal')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=fs)
        plt.tight_layout()
        save_path = os.path.join(savedir, f"stick_shape_{phase}")
        plt.savefig(f"{save_path}.png", dpi=400)
        plt.savefig(f"{save_path}.svg", dpi=400)
        plt.close(fig)

        all_arr[phase] = arr








def handedness(raw_data):
    mice = raw_data.keys()
    StartingPaws = pd.DataFrame(index=np.arange(0,160), columns=mice)
    TransitioningPaws = pd.DataFrame(index=np.arange(0,160), columns=mice)
    for midx, mouse in enumerate(mice):
        mouse_data = raw_data[mouse]
        initiating_limb = mouse_data.loc(axis=1)['initiating_limb']
        # Stack index level 0 'Day' together to remove this index
        initiating_limb = initiating_limb.droplevel('Day', axis=0)
        runstage_vals = initiating_limb.index.get_level_values('RunStage')
        runstage_series = pd.Series(runstage_vals, index=initiating_limb.index)

        # Find starting paws
        runstart_frames_mask = np.logical_and(runstage_series == 'RunStart',runstage_series.shift(1) == 'TrialStart').values
        runstart_frames = initiating_limb.index.get_level_values('FrameIdx')[runstart_frames_mask]

        starting = initiating_limb.loc(axis=0)[:,:,runstart_frames].droplevel('RunStage', axis=0).droplevel('FrameIdx', axis=0)
        starting_nums = starting.replace({'ForepawR': micestuff['LR']['ForepawToeR'], 'ForepawL': micestuff['LR']['ForepawToeL']})
        StartingPaws[mouse] = starting_nums

        # Find transitioning paws
        transition_frames_mask = np.logical_and(runstage_series == 'Transition', runstage_series.shift(1) == 'RunStart').values
        transition_frames = initiating_limb.index.get_level_values('FrameIdx')[transition_frames_mask]

        transitioning = initiating_limb.loc(axis=0)[:,:,transition_frames].droplevel('RunStage', axis=0).droplevel('FrameIdx', axis=0)
        # Replace 'R' with 1 and 'L' with 0
        transitioning_nums = transitioning.replace({'ForepawR': micestuff['LR']['ForepawToeR'], 'ForepawL': micestuff['LR']['ForepawToeL']})
        TransitioningPaws[mouse] = transitioning_nums

    savedir = os.path.join(r"H:\Characterisation", 'Handedness')
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    across_mice_L_R_percentage(TransitioningPaws, savedir)
    start_transition_consistency(StartingPaws, TransitioningPaws, savedir)

def plot_combo_sequence_per_mouse(StartingPaws, TransitioningPaws, savedir):
    import os
    import matplotlib.pyplot as plt

    combo_map = {
        (1, 1): 0,  # Left → Left
        (1, 2): 1,  # Left → Right
        (2, 1): 2,  # Right → Left
        (2, 2): 3  # Right → Right
    }
    combo_labels = [
        'Left start: Left transition',
        'Left start: Right transition',
        'Right start: Left transition',
        'Right start: Right transition'
    ]


    for mouse in StartingPaws.columns:
        start = StartingPaws[mouse]
        trans = TransitioningPaws[mouse]
        n_runs = len(start)

        combo_sequence = []
        valid_indices = []

        for i in range(n_runs):
            if pd.notna(start[i]) and pd.notna(trans[i]):
                key = (int(start[i]), int(trans[i]))
                if key in combo_map:
                    combo_sequence.append(combo_map[key])
                    valid_indices.append(i)

        # Plot
        fig, ax = plt.subplots(figsize=(5, 2.5))
        ax.step(valid_indices, combo_sequence, where='mid', color='black', linewidth=1)

        ax.set_yticks([0, 1, 2, 3])
        ax.set_yticklabels(combo_labels, fontsize=7)
        ax.set_ylim(-0.5, 3.5)
        ax.set_xlabel('Run', fontsize=7)
        ax.set_title(mouse, fontsize=7)
        ax.tick_params(axis='x', labelsize=7)
        ax.tick_params(axis='y', labelsize=7)

        plt.tight_layout()

        save_path = os.path.join(savedir, f"{mouse}_ComboStepPlot")
        plt.savefig(f"{save_path}.png", dpi=300)
        plt.savefig(f"{save_path}.svg", dpi=300)
        plt.close()

def start_transition_consistency(StartingPaws, TransitioningPaws, savedir):
    combo_data = []
    for mouse in StartingPaws.columns:
        df = pd.DataFrame({
            'Start': StartingPaws[mouse],
            'Transition': TransitioningPaws[mouse]
        }).dropna()

        df['Combo'] = df.apply(lambda row: f"{'Left' if row['Start'] == 1 else 'Right'} start: " +
                                           f"{'Left' if row['Transition'] == 1 else 'Right'} transition", axis=1)
        combo_counts = df['Combo'].value_counts(normalize=True)
        for combo, val in combo_counts.items():
            combo_data.append({'Mouse': mouse, 'Combo': combo, 'Proportion': val})

    combo_df = pd.DataFrame(combo_data)
    combo_pivot = combo_df.pivot(index='Mouse', columns='Combo', values='Proportion').fillna(0)

    # Define order and styling
    combo_order = [
        'Left start: Left transition',    # solid black
        'Left start: Right transition',   # black with white hatch
        'Right start: Left transition',   # white with black hatch
        'Right start: Right transition'   # solid white
    ]
    hatches = ['', '////', '////', '']
    facecolors = ['grey', 'grey', 'white', 'white']
    edgecolors = ['black', 'black', 'black', 'black']  # outline to contrast with fill

    fig, ax = plt.subplots(figsize=(5, 2))
    lefts = np.zeros(len(combo_pivot))

    for label, hatch, facecolor, edgecolor in zip(combo_order, hatches, facecolors, edgecolors):
        vals = combo_pivot[label].values if label in combo_pivot.columns else np.zeros(len(combo_pivot))
        ax.barh(combo_pivot.index, vals, left=lefts,
                color=facecolor, edgecolor=edgecolor, hatch=hatch, linewidth=1, label=label)
        lefts += vals

    ax.set_xlabel('Proportion of Runs', fontsize=7)
    ax.set_ylabel('Mouse', fontsize=7)
    ax.set_title('Start vs Transition Paw Combinations', fontsize=7)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1])
    ax.tick_params(axis='both', which='major', labelsize=7)
    ax.invert_yaxis()  # first mouse on top

    legend_elements = [
        Patch(facecolor=fc, edgecolor=ec, hatch=h, label=lbl, linewidth=1)
        for lbl, h, fc, ec in zip(combo_order, hatches, facecolors, edgecolors)
    ]
    ax.legend(handles=legend_elements, title='Combination',
              bbox_to_anchor=(1.05, 1), loc='upper left',
              frameon=False, fontsize=7, title_fontsize=7)

    plt.tight_layout()

    save_path = os.path.join(savedir, "StartTransition_PawCombination_perMouse")
    plt.savefig(f"{save_path}.png", dpi=300)
    plt.savefig(f"{save_path}.svg", dpi=300)
    plt.close()

def across_mice_L_R_percentage(TransitioningPaws, savedir):
    # Clean
    valid_trans = TransitioningPaws.dropna(how='all', axis=0).dropna(how='all', axis=1)
    rounded_trans = valid_trans.round().astype('Int64')

    # Long format
    df_long = rounded_trans.reset_index(drop=True).melt(var_name='Mouse', value_name='Paw')
    df_long = df_long.dropna()
    df_long['Paw'] = df_long['Paw'].map({1: 'Left', 2: 'Right'})

    # Per-mouse proportions
    mouse_counts = df_long.groupby(['Mouse', 'Paw']).size().unstack(fill_value=0)
    mouse_props = (mouse_counts.T / mouse_counts.sum(axis=1)).T.fillna(0)
    mouse_props_sorted = mouse_props.sort_index(ascending=False)

    # Set up plot
    fig, ax = plt.subplots(1, 1, figsize=(3, 2))

    # Custom stacked bars with solid fill and hatching
    left_vals = mouse_props_sorted['Left']
    right_vals = mouse_props_sorted['Right']
    y_pos = np.arange(len(mouse_props_sorted))

    # Plot left (solid black)
    ax.barh(y_pos, left_vals, color='black', edgecolor='black', label='Left')

    # Plot right (striped black) stacked on top
    ax.barh(y_pos, right_vals, left=left_vals, color='white', hatch='////', edgecolor='black', label='Right')

    # Axis formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels(mouse_props_sorted.index, fontsize=7)
    ax.set_xlim(0, 1)
    ax.set_xlabel('Proportion of Runs', fontsize=7)
    ax.set_ylabel('Mouse', fontsize=7)
    ax.tick_params(axis='both', which='major', labelsize=7)

    # Legend outside
    legend_elements = [
        Patch(facecolor='black', edgecolor='black', label='Left'),
        Patch(facecolor='white', edgecolor='black', hatch='////', label='Right')
    ]
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.0, 0.5),
              frameon=False, fontsize=7, title_fontsize=7)

    plt.tight_layout()
    save_path = os.path.join(savedir, f"TransitioningStride_perMouse_LR_proportions")
    plt.savefig(f"{save_path}.png", dpi=300)
    plt.savefig(f"{save_path}.svg", dpi=300)
    plt.close()











# Import data
with open(r"H:\\Characterisation\\LH_allpca_LhWnrm_res_-3-2-1_APA2Wash2\\preprocessed_data_APAChar_LowHigh.pkl",
          'rb') as f:
    data_LH = pickle.load(f)
print("Loaded data from preprocessed_data_APAChar_LowHigh.pkl")

with open(r"H:\Dual-belt_APAs\analysis\DLC_DualBelt\DualBelt_MyAnalysis\FilteredData\Round7_Jan25\APAChar_LowHigh\Extended\allmice.pickle",'rb') as file:
    raw_data = pickle.load(file)

indexes = data_LH['feature_data_notscaled'].loc(axis=0)[-1].index
# If idx has two levels: mouse and run
mouse_to_runs = {}
for mouse, run in indexes:
    mouse_to_runs.setdefault(mouse, []).append(run)

# Convert to np arrays if you want
mouse_to_runs = {mouse: np.array(runs) for mouse, runs in mouse_to_runs.items()}

angle_save_dir = r"H:\Characterisation_v2\Angles"
if not os.path.exists(angle_save_dir):
    os.makedirs(angle_save_dir)


plot_limb_positions_average(raw_data, mouse_to_runs, phases=['APA2', 'Wash2'], savedir=angle_save_dir, fs=7, n_interp=100)

plot_angles(data_LH['feature_data_notscaled'], phases=['APA2', 'Wash2'], stride=-1,
            savedir=angle_save_dir)
# handedness(raw_data)