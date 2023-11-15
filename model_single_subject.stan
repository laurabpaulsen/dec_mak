data {
  int<lower=0> n_trials;
  int<lower=1, upper=5> resp[n_trials]; // responses
  vector[n_trials] hit; // whether the answer was correct or not
  int<lower=1, upper=5> stim[n_trials]; // the sounds presented

  int<lower=1, upper=5> sounds[5];
  int<lower=1, upper=5> shapes[5];


}

parameters {
  real theta; // determinism in softmax function 
  real lr; // learning rate
  real lr_r; //reversal learning_rate
}

transformed parameters {
  matrix[5, 5] ass; // association strengths between pairs of sounds and shapes 
  ass[1,] = rep_row_vector(0.2, 5); ass[2,] = rep_row_vector(0.2, 5); ass[3,] = rep_row_vector(0.2, 5); ass[4,] = rep_row_vector(0.2, 5); ass[5,] = rep_row_vector(0.2, 5);
  matrix[5, n_trials] p; // softmax output
  p[,1] = rep_vector(0.2, 5);

  for (t in 2:n_trials) { // loop over trials
    
    if (hit[t-1] == 1) // if correct on t-1
      ass[stim[t-1], resp[t-1]] += lr; // NOTE: maybe ass should be 3d and then we would do t-1 here // update association strength of the stimuli and response pair
    else // if incorrect if incorrect on t-1
      ass[stim[t-1], resp[t-1]] -= lr; // update association strength of the stimuli and response pair

    // update associations for the sound chosen and remaining shapes not chosen
    for (shape in shapes){
      if (resp[t-1] != shape)
        if (hit[t-1] == 1)
          ass[stim[t-1], shape] -= lr_r;
        else
          ass[stim[t-1], shape] += lr_r;

    }
    // update the associations for sounds not played for the shape chosen
    for (sound in sounds){
      if (stim[t-1] != sound)
        if (hit[t-1] == 1)
          ass[sound, resp[t-1]] -= lr_r;
        else
          ass[sound, resp[t-1]] += lr_r;

    }
    
    p[,t] = softmax(theta * to_vector(ass[stim[t]]));

    }

}
model {
  theta ~ normal(1, 10);
  lr ~ normal(1, 10);
  lr_r ~ normal(1,10);


  for (t in 2:n_trials) {
    resp[t] ~ categorical(p[,t]);
  }

}