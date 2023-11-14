data {
  int<lower=0> n_trials;
  int<lower=1, upper=5> resp[n_trials]; // responses
  vector[n_trials] hit; // whether the answer was correct or not
  int<lower=1, upper=5> stim[n_trials]; // the sounds presented
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
      ass[stim[t-1], resp[t-1]] = ass[stim[t-1], resp[t-1]] + lr; // NOTE: maybe ass should be 3d and then we would do t-1 here // update association strength of the stimuli and response pair
    else // if incorrect if incorrect on t-1
      ass[stim[t-1], resp[t-1]] = ass[stim[t-1], resp[t-1]] -lr; // update association strength of the stimuli and response pair

    // IMPLEMENT UPDATING OF UNCHOSEN PARAMETERS

    
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