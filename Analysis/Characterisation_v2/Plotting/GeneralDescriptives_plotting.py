angle_nonscaled_stride = \
feature_data_compare_notscaled.loc(axis=1)[ 'signed_angle|ToeAnkle_ipsi_side_zref_swing_peak'].loc[-1]
mask_p1, mask_p2 = gu.get_mask_p1_p2(angle_nonscaled_stride, phases[0], phases[1])
angle_p1 = angle_nonscaled_stride[mask_p1]
angle_p2 = angle_nonscaled_stride[mask_p2]



# Group by run (assuming the run is on level=1) to compute the average across all observations per run
angle_p1_avg = angle_p1.groupby(level=1).mean()
angle_p2_avg = angle_p2.groupby(level=1).mean()
# Convert from degrees to radians as polar histograms require radians
theta_p1 = np.deg2rad(angle_p1_avg.values)
theta_p2 = np.deg2rad(angle_p2_avg.values)
# Define the bins for the polar histogram
# Here we create 20 equal bins spanning from 0 to 2*pi
num_bins = 360
bins = np.linspace(0, 2 * np.pi, num_bins + 1)
# Compute histograms: counts of run averages falling within each bin
hist_p1, _ = np.histogram(theta_p1, bins=bins)
hist_p2, _ = np.histogram(theta_p2, bins=bins)
# The width of each bin, needed for the bar plot
width = bins[1] - bins[0]
# Create a figure with two polar subplots side by side
fig, ax = plt.subplots(1, 1, subplot_kw={'projection': 'polar'}, figsize=(5, 5))
# Plot Phase 1: each bar's length corresponds to the number of runs in that angular bin
ax.bar(bins[:-1], hist_p1, width=width, bottom=0.0, align='edge', color='blue', alpha=0.6, label=p1)
ax.set_title("Phase 1: Run-Averaged Angle Distribution")
# Plot Phase 2
ax.bar(bins[:-1], hist_p2, width=width, bottom=0.0, align='edge', color='red', alpha=0.6, label=p2)
ax.set_title("Phase 2: Run-Averaged Angle Distribution")
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
ax.set_thetamin(180)
ax.set_thetamax(0)
plt.legend()
plt.show()