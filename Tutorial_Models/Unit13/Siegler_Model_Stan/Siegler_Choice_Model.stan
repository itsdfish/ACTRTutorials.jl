functions {
    #include ../Stan_Utilities/ACTR_Functions.stan

    // adjust base level activations so that chunks with sum < 5 have higher activation
    row_vector adjust_base_levels(row_vector activations, matrix result_values){
      row_vector[num_elements(activations)] new_activations;
      for(r in 1:rows(result_values)){
        new_activations[r] = activations[r];
        if(result_values[r,3] < 5){
          new_activations[r] += .65;
        }
      }
      return new_activations;
    }

    real computeLL(real d, real blc, real delta, real gamma, int k, real L, int N, real s, real tau, int sa, int mp, int bll,
      int n_chunks, int n_slots, matrix memory_slots, matrix memory_values, row_vector stimulus_slots, row_vector stimulus_values, int choice,
      int choice_rep){
      row_vector[n_slots] imaginal_slots;
      row_vector[n_slots] imaginal_values;
      row_vector[n_chunks+1] activations; // n_chunks + threshold
      row_vector[1] response_slots;
      row_vector[1] response_values;
      // slots: request_result[1,1]
      // values: request_result[1,2]
      // number of chunks: request_result[1,3][1,1]
      matrix[n_chunks,n_slots] request_result[1,3];
      matrix[n_chunks,n_slots] result_slots;
      matrix[n_chunks,n_slots] result_values;
      real lags[1];
      real denoms[n_slots];
      int n_results;
      real prob;

      // Return chunks that match retrieval request (stimulus_slots, stimulus_values). If mp = 1,
      // as in the current case, all chunk slots and values are returned
      request_result = request_chunks(stimulus_slots, stimulus_values, memory_slots, memory_values, mp);
      result_slots = request_result[1,1];
      result_values = request_result[1,2];
      
      n_results = to_int(request_result[1,3][1,1]);
      // compute activation for the result of the retrieval request
      activations = compute_all_activations(blc, d, delta, gamma, tau, k, L, N, lags, sa, mp, bll, result_slots, 
        result_values, stimulus_slots, stimulus_values, n_results, n_chunks, imaginal_slots, imaginal_values, denoms);
      // adjust base level activations so that chunks with sum < 5 have higher activation
      activations = adjust_base_levels(activations, result_values);

      // log likelihood for retrieval failure
      if(choice == -100){
        prob = compute_retrieval_prob(activations, s, tau, n_chunks, n_results + 1);
        return choice_rep * log(prob);
      }
      // third slot index corresponds to sum
      response_slots[1] = 3;
      // slot-value is the sum or given response
      response_values[1] = choice;
      // computes the marginal probability of the response. If choice = 3, then the probability of retrieving 1 + 2, plus
      // the probability of retrieving 2 + 1, and so on...
      prob = marginal_retrieval_prob(activations, s, tau, result_slots, result_values, response_slots, 
        response_values, n_results);
      // compute log probability and multiply by the number of times this response was given to the stimulus
      return choice_rep * log(prob);
    }
}

data {
  int<lower=1> n_obs;                // number of observations
  int<lower=-100> choice[n_obs];     // response indices
  int choice_reps[n_obs];            // number of times response was made
  int<lower=1> n_slots;
  int<lower=1> n_chunks;
  matrix[n_obs,2] stimuli_slots;     //num1,num2
  matrix[n_obs,2] stimuli_values;    //1,2 etc
  matrix[n_chunks,3] memory_values;  //num1,num2,sum
  matrix[n_chunks,3] memory_slots;   //1,2,3 etc
  int<lower=0,upper=1> mp;           // mismatch penality on
  int<lower=0,upper=1> bll;          // baselevel learning on
  int<lower=0,upper=1> sa;           // spreading activation on
}

parameters {
  real delta;                   // mismatch penality
  real tau;                     // retrieval threshold
  real<lower=0> s;              // logistic scale
}

model {
    // all parameters declared here are not used by the model, but must be initialized
    // and passed to functions 

    // decay
    real d;
    // baselevel activation parameter
    real blc;
    // number of uses tracks
    int k;
    // chunk life time
    real L;
    // number of uses of chunk
    int N;
    // spreading activation parameter 
    real gamma;

    d = 0.0;
    gamma = 0.0;
    blc = 0.0;
    k = 1;
    L = 0;
    N = 0;


    delta ~ normal(2.5, 1);       // prior for base level constant
    tau ~ normal(-.45, .5);            // prior for retrieval threshold
    s ~ normal(.5, .5);              // prior for logistic scale

    // loop over data and compute log likelihood of each response
    for(trial in 1:n_obs){
      target += computeLL(d, blc, delta, gamma, k, L, N, s, tau, sa, mp, bll, n_chunks, n_slots,
         memory_slots, memory_values, stimuli_slots[trial,:], stimuli_values[trial,:], choice[trial], 
         choice_reps[trial]);
    }
}
