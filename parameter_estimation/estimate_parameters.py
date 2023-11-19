"""
This script holds the code for estimating the parameters.
"""
import pandas as pd

def prep_data(data:pd.Dataframe):
    """
    Prepares the experimental data for the parameter estimation using Stan.
    """
    


    """
    # FROM PARAMETER_RECOVERY
    stan_data = {
        "n_trials": len(response),
        "stim": [int(stim) + 1 for stim in stimuli], # stan starts counting at 1 instead of 0
        "resp": [int(resp) + 1 for resp in response],
        "hit": hit,
        "sounds": list(range(1,6)),
        "shapes": list(range(1,6)),
    }
    """

    return stan_data





def main():
    pass


if __name__ == "__main__":
    main()