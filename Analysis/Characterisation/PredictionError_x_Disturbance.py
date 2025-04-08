import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load data
with open(r"H:\Dual-belt_APAs\Plots\Jan25\FeatureReduction\Round23-9mice_ManualFeatRed_NoSelection-c=1\global_data_APAChar_LowHigh.pkl", "rb") as f:
    apa_data = pickle.load(f)
with open(r"H:\Dual-belt_APAs\Plots\Jan25\FeatureReduction\Round23-9mice_ManualFeatRed_Perturbation_NoSelection-c=1\global_data_APAChar_LowHigh.pkl", "rb") as f:
    disturbance_data = pickle.load(f)

apa_pred = apa_data["aggregated_predictions"]
dist_pred = disturbance_data["aggregated_predictions"]

# Define target values and x values
target_apa = np.ones(100)
target_wash = np.zeros(50)
target = np.concatenate((target_apa, target_wash))
target_x_vals = np.arange(0, 160, 1)

# Process each key in apa_pred
for (p1, p2, stride_number) in apa_pred.keys():
    apa_pred_mice = apa_pred[(p1, p2, stride_number)]
    dist_pred_mice = dist_pred[('APA1', 'APA2', 0)]

    # Create dictionaries to store prediction error and disturbance data for each mouse
    pred_error_dict = {}
    disturbance_dict = {}

    for i in range(len(apa_pred_mice)):
        apa_mouseid = apa_pred_mice[i].mouse_id
        apa_x_vals = np.array(apa_pred_mice[i].x_vals)
        apa_smoothed_scaled_pred = apa_pred_mice[i].smoothed_scaled_pred

        # Find corresponding mouse in disturbance data
        for j in range(len(dist_pred_mice)):
            if dist_pred_mice[j].mouse_id == apa_mouseid:
                dist_x_vals = np.array(dist_pred_mice[j].x_vals)
                dist_smoothed_scaled_pred = dist_pred_mice[j].smoothed_scaled_pred
                break

        # Determine available x values
        available_apa_mask = np.isin(target_x_vals, apa_x_vals)
        available_dist_mask = np.isin(target_x_vals, dist_x_vals)

        # Create ordered arrays with NaNs for missing values
        ordered_apa_pred = np.full(len(available_apa_mask), np.nan)
        ordered_apa_pred[available_apa_mask] = apa_smoothed_scaled_pred
        ordered_dist_pred = np.full(len(available_dist_mask), np.nan)
        ordered_dist_pred[available_dist_mask] = dist_smoothed_scaled_pred

        # Trim (exclude first 10 values)
        selected_apa_pred = ordered_apa_pred[10:]
        selected_dist_pred = ordered_dist_pred[10:]
        selected_target_x_vals = target_x_vals[10:]

        # Calculate prediction error and scale values
        prediction_error_apa = abs(target - selected_apa_pred)
        scaled_prediction_error_apa = prediction_error_apa / np.nanmax(abs(prediction_error_apa))
        scaled_disturbance_prediction = selected_dist_pred / np.nanmax(abs(selected_dist_pred))

        pred_error_dict[apa_mouseid] = scaled_prediction_error_apa
        disturbance_dict[apa_mouseid] = scaled_disturbance_prediction

    # ---------------------------
    # Plot individual mouse regressions in subplots
    # ---------------------------
    mice = list(pred_error_dict.keys())
    n_mice = len(mice)
    n_cols = 3
    n_rows = (n_mice + n_cols - 1) // n_cols  # e.g., 3 rows for 9 mice
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 15))
    axs = axs.flatten()

    # Lists to store data for overall regression
    all_x_list = []
    all_y_list = []

    for i, mouse in enumerate(mice):
        # Get the first 100 data points and remove NaNs
        x = pred_error_dict[mouse].reshape(-1, 1)[:100]
        y = disturbance_dict[mouse].reshape(-1, 1)[:100]
        mask = ~np.isnan(x.flatten()) & ~np.isnan(y.flatten())
        x_clean = x[mask].reshape(-1, 1)
        y_clean = y[mask].reshape(-1, 1)

        # Append to overall lists
        all_x_list.append(x_clean)
        all_y_list.append(y_clean)

        # Fit regression for this mouse
        reg = LinearRegression().fit(x_clean, y_clean)
        slope = reg.coef_[0][0]
        intercept = reg.intercept_[0]
        r_squared = reg.score(x_clean, y_clean)
        y_pred = reg.predict(x_clean)

        # Plot on the corresponding subplot
        ax = axs[i]
        ax.plot(x_clean, y_clean, 'o', label=mouse)
        ax.plot(x_clean, y_pred, label='Regression')
        ax.set_xlabel('Prediction Error')
        ax.set_ylabel('Disturbance Prediction')
        ax.set_title(f'{mouse}')

        # Annotate regression stats
        # Determine position at right end of regression line
        x_max = np.max(x_clean)
        y_max = reg.predict([[x_max]])[0][0]

        stats_text = f'Slope: {slope:.2f}\nIntercept: {intercept:.2f}\n$R^2$: {r_squared:.2f}'
        ax.text(x_max, y_max, stats_text, horizontalalignment='left', verticalalignment='center',
                fontsize=10)#, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.legend()

    # Hide any extra subplots if there are fewer than grid cells
    for j in range(i + 1, len(axs)):
        axs[j].set_visible(False)

    plt.tight_layout()
    plt.show()

    plt.savefig(r"H:\Dual-belt_APAs\Plots\Jan25\FeatureReduction\Round23-9mice_PredErrorXDisturbance\IndivdualMouseRegression.png")
    plt.close()

    # ---------------------------
    # Overall regression using all mice's data
    # ---------------------------
    all_x = np.concatenate(all_x_list)
    all_y = np.concatenate(all_y_list)
    reg_all = LinearRegression().fit(all_x, all_y)
    slope_all = reg_all.coef_[0][0]
    intercept_all = reg_all.intercept_[0]
    r_squared_all = reg_all.score(all_x, all_y)

    # Plot combined scatter and overall regression line
    fig, ax = plt.subplots(figsize=(6, 12))
    ax.scatter(all_x, all_y, alpha=0.5, label='All data')
    # Create a sorted x range for plotting the line
    x_line = np.linspace(np.nanmin(all_x), np.nanmax(all_x), 100).reshape(-1, 1)
    y_line = reg_all.predict(x_line)
    ax.plot(x_line, y_line, 'k--', linewidth=2, label='Overall Regression')
    stats_text_all = f'Slope: {slope_all:.2f}\nIntercept: {intercept_all:.2f}\n$R^2$: {r_squared_all:.2f}'
    ax.text(0.05, 0.95, stats_text_all, transform=ax.transAxes, fontsize=12,
            verticalalignment='top')#, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.set_xlabel('Prediction Error')
    ax.set_ylabel('Disturbance Prediction')
    ax.set_title('Overall Regression for All Mice')
    plt.tight_layout()
    plt.show()

    plt.savefig(r"H:\Dual-belt_APAs\Plots\Jan25\FeatureReduction\Round23-9mice_PredErrorXDisturbance\AllMiceRegression.png")
    plt.close()
