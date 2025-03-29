import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Helpers.Config_23 import *

def ShufflingTest_ComparePhases(Obs_p1, Obs_p2, meanObs_p1, meanObs_p2, phase1, phase2, num_iter=1000) -> [float, float, np.array]:
    """
    Shuffling test to determine if the difference in means between two phases is significant.
    Args:
        Obs_p1: DataFrame, Observations for phase1 for all trials and mice
        Obs_p2: DataFrame, Observations for phase2 for all trials and mice
        meanObs_p1: DataFrame, Mean observations for phase1 for each mouse
        meanObs_p2: DataFrame, Mean observations for phase2 for each mouse
        phase1: str, Name of phase1
        phase2: str, Name of phase2
        num_iter: int, Number of iterations for the shuffling test
    Returns:
        p_value: float, p-value of the shuffling test
        Eff_obs: float, Observed effect size
        EffNull: array, Null effect size distribution
    """

    Eff_obs = (meanObs_p2 - meanObs_p1).mean()

    # Identify phase1 and phase2 trials
    phase1_trials = expstuff['condition_exp_runs']['APAChar']['Extended'][phase1]
    phase2_trials = expstuff['condition_exp_runs']['APAChar']['Extended'][phase2]
    all_trials = np.concatenate((phase1_trials, phase2_trials))

    Obs_both = pd.concat([Obs_p1, Obs_p2], axis=0)

    Obs_matrix = Obs_both.unstack(level='Run').loc[:, all_trials].values
    trial2idx = {trial: i for i, trial in enumerate(all_trials)}

    # Initialise null effect Eff^Null{numiter x 1}
    EffNull = np.empty(num_iter)

    # For  iter in numiter:
    for iter in range(num_iter):
        # Randomly select null phase1 and phase2 trials to make phase1null and phase2null
        phase1_trials_null = np.random.choice(all_trials, len(phase1_trials), replace=False)
        phase2_trials_null = all_trials[~np.isin(all_trials, phase1_trials_null)]
        phase2_trials_null = np.random.choice(phase2_trials_null, len(phase2_trials), replace=False)

        idx1 = [trial2idx[t] for t in phase1_trials_null]
        idx2 = [trial2idx[t] for t in phase2_trials_null]

        # Get means of each mouse for phase1null and phase2null
        mouse_means_p1 = np.nanmean(Obs_matrix[:, idx1], axis=1)
        mouse_means_p2 = np.nanmean(Obs_matrix[:, idx2], axis=1)

        # Calculate the difference in means for each mouse
        EffNull[iter] = np.nanmean(mouse_means_p2 - mouse_means_p1)

    # Count how many values in Eff^Null are greater than Eff^Obs
    # Two‑tailed p‑value (adding +1 to numerator and denominator for a small‐sample correction)
    p_value = (np.sum(np.abs(EffNull) >= abs(Eff_obs)) + 1) / (num_iter + 1)

    return p_value, Eff_obs, EffNull

def ShufflingTest_CompareConditions(Obs_p1, Obs_p2 ,Obs_p1c ,Obs_p2c ,pdiff_Obs ,pdiff_c_Obs, phase1, phase2, num_iter=1000):
    Eff_obs = (pdiff_Obs - pdiff_c_Obs).mean()

    # Identify phase1 and phase2 trials
    phase1_trials = expstuff['condition_exp_runs']['APAChar']['Extended'][phase1]
    phase2_trials = expstuff['condition_exp_runs']['APAChar']['Extended'][phase2]
    all_trials = np.concatenate((phase1_trials, phase2_trials))

    # Ensure inputs are Series
    Obs_p1 = Obs_p1.squeeze()
    Obs_p2 = Obs_p2.squeeze()
    Obs_p1c = Obs_p1c.squeeze()
    Obs_p2c = Obs_p2c.squeeze()

    # Build trial×mouse matrices for each condition
    Obs_matrix = pd.concat([Obs_p1, Obs_p2], axis=0).unstack(level='Run').loc[:, all_trials].values
    Obs_matrixc = pd.concat([Obs_p1c, Obs_p2c], axis=0).unstack(level='Run').loc[:, all_trials].values

    trial2idx = {trial: i for i, trial in enumerate(all_trials)}
    EffNull = np.empty(num_iter)

    for i in range(num_iter):
        phase1_null = np.random.choice(all_trials, len(phase1_trials), replace=False)
        phase2_pool = all_trials[~np.isin(all_trials, phase1_null)]
        phase2_null = np.random.choice(phase2_pool, len(phase2_trials), replace=False)

        idx1 = [trial2idx[t] for t in phase1_null]
        idx2 = [trial2idx[t] for t in phase2_null]

        eff = np.nanmean(Obs_matrix[:, idx2], axis=1) - np.nanmean(Obs_matrix[:, idx1], axis=1)
        effc = np.nanmean(Obs_matrixc[:, idx2], axis=1) - np.nanmean(Obs_matrixc[:, idx1], axis=1)
        EffNull[i] = np.nanmean(eff - effc)

    p_value = (np.sum(np.abs(EffNull) >= abs(Eff_obs)) + 1) / (num_iter + 1)
    return p_value, Eff_obs, EffNull






