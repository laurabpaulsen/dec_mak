"""
Loops over n_subjects and recovers the parameters for each subject.

dev notes:
- [ ] Figure out which distributions to use to draw the parameters from
"""

import stan
from generate import experimental_loop
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_recovery_ax(ax, true, estimated, parameter_name):
    """
    Helper function for plot_recoveries
    """
    ax.scatter(true, estimated)
    x_lims = ax.get_xlim()
    ax.plot([0, x_lims[1]], [0, x_lims[1]], color = "black", linestyle = "dashed")
    ax.set_xlabel("True")
    ax.set_ylabel("Estimated")
    ax.set_title(parameter_name.title())


def plot_recoveries(trues:list, estimateds:list, parameter_names:list, savepath:Path):
    """
    Plot the recovery of the parameters.

    Parameters
    ----------
    trues : list
        List of true parameters.
    estimateds : list
        List of estimated parameters.
    parameter_names : list
        List of parameter names.
    savepath : Path
        Path to save the figure to.
    
    Returns
    -------
    None
    """

     # plot true vs estimated parameters
    fig, axes = plt.subplots(1, len(trues), figsize = (15, 5))
    
    for true, estimated, parameter_name, axis in zip(trues, estimateds, parameter_names, axes):
        plot_recovery_ax(axis, true, estimated, parameter_name)

    plt.tight_layout()
    
    if savepath:
        plt.savefig(savepath / "subject_parameter_recovery.png")


def test_parameter_recovery(n_subjects, model_spec, savepath = None):
    """
    Generate synthetic data and fit the model to it. Check how well the parameters are recovered by plotting median against the true parameters.

    Parameters
    ----------
    n_subjects : int
        Number of subjects.
    model_spec : str
        Stan model specification.
    savepath : Path, optional
        Path to save the parameter recovery figure to, by default None
    """
    # true theta, learning rate, reversal learning rate
    t_theta, t_lr, t_rlr = np.zeros((n_subjects)), np.zeros((n_subjects)), np.zeros((n_subjects))

    # estimated theta, learning rate, reversal learning rate
    e_theta, e_lr, e_rlr= np.zeros((n_subjects)), np.zeros((n_subjects)), np.zeros((n_subjects))

    for subject in range(n_subjects):

        # choose random parameters UPDATE THIS
        theta = np.random.poisson(5)

        # lr and rlr should be between 0 and 1 with more values closer to 0
        lr = np.random.beta(2, 30)
        rlr = np.random.beta(2, 30)

        # generate synthetic data
        _, stimuli, response, hit = experimental_loop(
            n_trials = 100, 
            theta = theta,
            learning_rate = lr, # learning rate for chosen nodes
            reversal_learning_rate = rlr # learning rate for non-chosen nodes
        )

        # prepare data for Stan
        data = {
            "n_trials": len(response),
            "stim": [int(stim) + 1 for stim in stimuli], # stan starts counting at 1 instead of 0
            "resp": [int(resp) + 1 for resp in response],
            "hit": hit,
            "sounds": list(range(1,6)),
            "shapes": list(range(1,6)),
        }

        model = stan.build(model_spec, data=data)

        fit = model.sample(num_chains=4, num_samples=1000)

        df = fit.to_frame()  # pandas `DataFrame, requires pandas

        # save true and e parameters
        t_theta[subject] = theta
        t_lr[subject] = lr
        t_rlr[subject] = rlr

        e_theta[subject] = df['theta'].median()
        e_lr[subject] = df['lr'].median()
        e_rlr[subject] = df['lr_r'].median()
    
    # plot true vs estimated parameters
    plot_recoveries(
        trues = [t_theta, t_lr, t_rlr],
        estimateds = [e_theta, e_lr, e_rlr],
        parameter_names = ["theta", "learning_rate", "reversal_learning_rate"],
        savepath = savepath
    )
   

def main():
    path = Path(__file__).parent

    outpath = path / "fig"

    if not outpath.exists():
        outpath.mkdir()

    with open(path.parent / "single_subject.stan") as f:
        model_spec = f.read()
    
    n_subjects = 50

    test_parameter_recovery(n_subjects, model_spec, savepath = outpath)


if __name__ in "__main__":
    main()