"""
"""

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from pathlib import Path

def softmax(x, theta = 1):
    """
    Compute softmax values for each sets of scores in x

    Parameters
    ----------
    x : numpy array
        Array of values to apply softmax to
    inverse_heat : float
        Inverse temperature for the softmax. The default is 1.

    Returns
    -------
    softmax_x : numpy array
        Softmaxed values
    """

    # Joshs r code
    # exp_p[t,] <- exp(theta*w[t,stim[t],])
    # for (i in 1:nmel) {
    #       p[t,i] <- exp_p[t,i]/sum(exp_p[t,])
    # } 
    
    # get the exponentiated values
    exp_x = np.exp(theta * x)

    # get the softmaxed values
    softmax_x = exp_x / np.sum(exp_x)

    return softmax_x



def associations_at_start(n_shapes = 5, n_sounds = 5):
    """ 
    Generates pandas dataframe with a uniform representation for the pairwise association between shapes and sounds
    """

    uniform_association = 1/n_shapes

    # empty pandas dataframe one column for each shape and one row for each sound
    df = pd.DataFrame(index = range(n_sounds), columns = range(n_shapes))

    # fill dataframe with uniform association
    df = df.fillna(uniform_association)

    return df


def update_associations(sound_played:int, shape_chosen:int, feedback:bool, associations:pd.DataFrame, learning_rate = 0.05, reversal_learning_rate = 0.05, n_shapes = 5, n_sounds = 5):
    """
    Updates the associations between shapes and sounds based on the feedback received
    
    Parameters
    ----------
    sound_played : int
        Index of the sound played
    shape_chosen : int
        Index of the shape chosen
    feedback : bool
        Feedback received for the choice made (True = correct, False = incorrect)
    associations : pandas dataframe
        Associations between shapes and sounds
    
    Returns
    -------
    None. The associations dataframe is updated in place
    """

    # update the chosen shape-sound association
    if feedback: # if correct, increase the association
        associations.loc[sound_played, shape_chosen] += learning_rate
    else: # if incorrect, decrease the association
        associations.loc[sound_played, shape_chosen] -= learning_rate
    
    # update the associations for shapes not chosen for the sound played
    for i in range(n_shapes): 
        if i != shape_chosen:
            if feedback:
                associations.loc[sound_played, i] -= reversal_learning_rate 
            else:
                associations.loc[sound_played, i] += reversal_learning_rate

    # update the associations for sounds not played for the shape chosen
    for i in range(n_sounds):
        if i != sound_played:
            if feedback:
                associations.loc[i, shape_chosen] -= reversal_learning_rate
            else:
                associations.loc[i, shape_chosen] += reversal_learning_rate

def play_sound(n_sounds = 5):
    """
    Chooses a sound to play at random
    """

    return random.randint(0, n_sounds - 1)

def choose_a_shape(associations:pd.DataFrame, sound_played:int, theta = 1):
    """
    Chooses a shape to play based on the associations between shapes and sounds
    """

    # get the associations for the sound played
    sound_associations = associations.loc[sound_played, :]

    # apply softmax to the associations
    sound_associations_softmax = softmax(sound_associations, theta = theta)

    # choose a shape based on the softmaxed associations
    shape_chosen = np.random.choice(range(len(sound_associations)), p = sound_associations_softmax)

    return shape_chosen

def experimental_loop(n_trials, sound_shape_mapping = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}, theta = 2, learning_rate = 0.05, reversal_learning_rate = 0.05, n_shapes = 5, n_sounds = 5):
    """

    Parameters
    ----------
    n_trials : int
        Number of trials to run
    sound_shape_mapping : dict
        Dictionary mapping sounds to shapes where the keys are the sounds and the values are the shapess
    theta : float
        Inverse temperature for the softmax
    learning_rate : float, optional
        Learning rate for updating the associations. The default is 0.05.
    n_shapes : int, optional
        Number of shapes. The default is 5.
    n_sounds : int, optional
        Number of sounds. The default is 5.
    
    Returns
    -------
    associations_list : list
        List of pandas dataframes with the associations between shapes and sounds at each trial
    """
    stimuli = np.zeros((n_trials))
    response = np.zeros((n_trials))
    hit = np.zeros((n_trials))

    # get uniform associations at start
    associations = associations_at_start()

    # list to save associations at each trial
    associations_list = [associations.copy()]


    # loop over trials
    for trial in range(n_trials):
        sound = play_sound()
        stimuli[trial] = sound

        shape = choose_a_shape(associations, sound, theta = theta)

        feedback = sound_shape_mapping[sound] == shape

        update_associations(
            sound, 
            shape, 
            feedback, 
            associations, 
            learning_rate = learning_rate, 
            reversal_learning_rate = reversal_learning_rate)

        associations_list.append(associations.copy())

        response[trial] = shape
        hit[trial] = int(feedback)
    
    return associations_list, stimuli, response, hit


def main():
    path = Path(__file__).parent

    fig_path = path / "fig"

    # ensure that the fig folder exists
    if not fig_path.exists():
        fig_path.mkdir()


    sound_shape_mapping = {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 4
    }

    associations_list, response, hit = experimental_loop(
        n_trials=100, 
        sound_shape_mapping = sound_shape_mapping,
        theta = 2,
        learning_rate = 0.05, # learning rate for chosen nodes
        reversal_learning_rate = 0.01 # learning rate for non-chosen nodes
        )
    
    # save dataframe with response and hit
    df = pd.DataFrame({"response": response, "hit": hit})
    df.to_csv("data.tsv", sep = "\t", index = False)


    # plot how the associations change over time for each sound
    fig, axes = plt.subplots(1, 5, figsize = (20, 5), sharey=True)

    for sound in range(5):
        tmp_associations = [associations.loc[sound, :] for associations in associations_list]
        axes[sound].plot(tmp_associations)
        axes[sound].set_title(f"Sound {sound}")
    
    for ax in axes:
        ax.set_xlabel("Trial")
        ax.set_ylabel("Association")


    plt.savefig(fig_path / "associations.png")

    # plot the choices made over time and correct/incorrect feedback
    fig, ax = plt.subplots(1, 1, figsize = (20, 5))

    ax.plot(response, label = "Response")
    ax.plot(hit, label = "Hit")
    ax.legend()

    ax.set_xlabel("Trial")
    
    plt.savefig(fig_path / "choices.png")

if __name__ in "__main__":    
    main()