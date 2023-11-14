import pystan
from generate_synthetic_data import experimental_loop
from pathlib import Path

if __name__ in "__main__":
    path = Path(__file__).parent
    shape_sound_mapping = {
        0: 3,
        1: 2,
        2: 1,
        3: 0,
        4: 4
    }

    # generate synthetic data
    _, response, hit = experimental_loop(
        n_trials=100, 
        sound_shape_mapping = shape_sound_mapping,
        theta = 2,
        learning_rate = 0.05, # learning rate for chosen nodes
        reversal_learning_rate = 0.01 # learning rate for non-chosen nodes
        )
    
    model = pystan.StanModel(path / "model.stan")

    # model specificatio


    # prepare data for Stan
    data = {
        "n_trials": len(response),
        "response": int(response),
        "hit": int(hit)
    }
    
    #  fit model
    fit = model.sampling(data=data, iter=1000, chains=4)

    # extract results
    results = fit.extract()
    print(results)
    

    
