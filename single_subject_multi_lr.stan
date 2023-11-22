data {
  int<lower=0> n_trials;
  array[n_trials] int<lower=1, upper=5> resp; // responses
  array[n_trials] int<lower=1, upper=5> stim; // the sounds presented
  vector[n_trials] hit; // whether the answer was correct or not
  array[5] int<lower=1, upper=5> sounds; // identifier for each sound
  array[5] int<lower=1, upper=5> shapes; // identifier for each shape
}

parameters {
  real <lower=0> theta; // determinism in softmax function 
  real <lower=0, upper=1> lr; // learning rate
  real <lower=0, upper=1> lr_r_shape; //reversal learning_rate shape
  real <lower=0, upper=1> lr_r_sound; //reversal learning_rate sound
}

transformed parameters {
  matrix[5, 5] ass; // association strengths between pairs of sounds and shapes 
  ass[1,] = rep_row_vector(0.2, 5); ass[2,] = rep_row_vector(0.2, 5); ass[3,] = rep_row_vector(0.2, 5); ass[4,] = rep_row_vector(0.2, 5); ass[5,] = rep_row_vector(0.2, 5);
  matrix[5, n_trials] p; // softmax output
  p[,1] = rep_vector(0.2, 5);

  for (t in 2:n_trials) { // loop over trials
    
    if (hit[t-1] == 1) // if correct on t-1
      ass[stim[t-1], resp[t-1]] += lr; // update association strength of the stimuli and response pair
    else // if incorrect if incorrect on t-1
      ass[stim[t-1], resp[t-1]] -= lr; // update association strength of the stimuli and response pair

    // update associations for the sound chosen and remaining shapes not chosen
    for (shape in shapes){
      if (resp[t-1] != shape)
        if (hit[t-1] == 1)
          ass[stim[t-1], shape] -= lr_r_sound;
        else
          ass[stim[t-1], shape] += lr_r_sound;

    }
    // update the associations for sounds not played for the shape chosen
    for (sound in sounds){
      if (stim[t-1] != sound)
        if (hit[t-1] == 1)
          ass[sound, resp[t-1]] -= lr_r_shape;
        else
          ass[sound, resp[t-1]] += lr_r_shape;

    }
    
    p[,t] = softmax(theta * to_vector(ass[stim[t]]));

    }

}
model {
  // Figure out more appropriate priors
  // Currently truncating everything at 0
  // theta ~ normal(1, 10) T[0, ];
  // lr ~ normal(1, 10) T[0, ];
  // lr_r ~ normal(1,10) T[0, ];

  theta ~ normal(1, 10) T[0, ];
  lr ~ normal(.5, .5);
  lr_r_sound ~ normal(.5, .5);
  lr_r_shape ~ normal(.5, .5);

  for (t in 2:n_trials) {
    resp[t] ~ categorical(p[,t]);
  }

}