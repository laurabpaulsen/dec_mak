import stan
from generate_synthetic_data import experimental_loop
from pathlib import Path

if __name__ in "__main__":
    path = Path(__file__).parent
    
    shape_sound_mapping = {
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        0: 0
        }


    # generate synthetic data
    _, stimuli, response, hit = experimental_loop(
        n_trials=100, 
        sound_shape_mapping = shape_sound_mapping,
        theta = 2,
        learning_rate = 0.05, # learning rate for chosen nodes
        reversal_learning_rate = 0.01 # learning rate for non-chosen nodes
        )
    
    # prepare data for Stan
    data = {
        "n_trials": len(response),
        "stim": [int(stim) + 1 for stim in stimuli], 
        "resp": [int(resp) + 1 for resp in response],
        "hit": hit
    }

    with open(path / "model_single_subject.stan") as f:
        model_spec = f.read()
    
    model = stan.build(model_spec, data=data)

    fit = model.sample(num_chains=4, num_samples=1000)

    df = fit.to_frame()  # pandas `DataFrame, requires pandas
    print(f"Mean theta: {df['theta'].mean()}")
    print(f"Mean lr: {df['lr'].mean()}")
    print(f"Mean lr_r: {df['lr_r'].mean()}")
    

    
