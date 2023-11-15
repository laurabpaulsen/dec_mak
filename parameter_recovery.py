import stan
from generate_synthetic_data import experimental_loop
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def test_parameter_recovery(n_subjects, model_spec, savepath = None):
    """
    Generate synthetic data and fit the model to it. Check how well the parameters are recovered by plotting median against the true parameters.
    """

    true_theta = np.zeros((n_subjects))
    true_learning_rate = np.zeros((n_subjects))
    true_reversal_learning_rate = np.zeros((n_subjects))

    estimated_theta = np.zeros((n_subjects))
    estimated_learning_rate = np.zeros((n_subjects))
    estimated_reversal_learning_rate = np.zeros((n_subjects))

    for subject in range(n_subjects):

        # choose random parameters
        theta = np.random.uniform(0, 5)
        learning_rate = np.random.uniform(0, 1)
        reversal_learning_rate = np.random.uniform(0, 1)

        # generate synthetic data
        _, stimuli, response, hit = experimental_loop(
            n_trials=100, 
            theta = theta,
            learning_rate = learning_rate, # learning rate for chosen nodes
            reversal_learning_rate = reversal_learning_rate # learning rate for non-chosen nodes
        )

        # prepare data for Stan
        data = {
            "n_trials": len(response),
            "stim": [int(stim) + 1 for stim in stimuli], 
            "resp": [int(resp) + 1 for resp in response],
            "hit": hit,
            "sounds": list(range(1,6)),
            "shapes": list(range(1,6)),
        }

        model = stan.build(model_spec, data=data)

        fit = model.sample(num_chains=4, num_samples=1000)

        df = fit.to_frame()  # pandas `DataFrame, requires pandas

        # save true and estimated parameters
        true_theta[subject] = theta
        true_learning_rate[subject] = learning_rate
        true_reversal_learning_rate[subject] = reversal_learning_rate

        estimated_theta[subject] = df['theta'].median()
        estimated_learning_rate[subject] = df['lr'].median()
        estimated_reversal_learning_rate[subject] = df['lr_r'].median()
    
    # plot true vs estimated parameters
    fig, ax = plt.subplots(1, 3, figsize = (15, 5))
    
    for true, estimated, parameter_name, axis in zip(
        [true_theta, true_learning_rate, true_reversal_learning_rate],
        [estimated_theta, estimated_learning_rate, estimated_reversal_learning_rate],
        ["theta", "learning rate", "reversal learning rate"],
        ax
    ):
        plot_recovery_ax(axis, true, estimated, parameter_name)

    plt.tight_layout()
    
    if savepath:
        plt.savefig("parameter_recovery.png")


def plot_recovery_ax(ax, true, estimated, parameter_name):
    ax.scatter(true, estimated)
    ax.plot([0, 1], [0, 1], color = "black", linestyle = "dashed")
    ax.set_xlabel("True")
    ax.set_ylabel("Estimated")
    ax.set_title(parameter_name.title())
    

if __name__ in "__main__":
    path = Path(__file__).parent

    with open(path / "model_single_subject.stan") as f:
        model_spec = f.read()
    
    n_subjects = 50

    test_parameter_recovery(n_subjects, model_spec, savepath = path / "parameter_recovery.png")

    

    
