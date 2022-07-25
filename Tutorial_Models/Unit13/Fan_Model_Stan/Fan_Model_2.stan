functions {
  #include ../Stan_Utilities/ACTR_Functions.stan

  real LLYes(real rt,real sigma,real ter, row_vector request_slots, row_vector request_values, matrix result_slots,matrix result_values,
      row_vector activations){
        int idx;
        // get index corresponding to retrieved chunk
        idx = get_chunk_index(result_slots, result_values, request_slots, request_values);
        // log normal race log likelihood of response idx at rt
        return LNR_LL(-activations, sigma, ter, rt, idx);
    }

  real LLNo(real rt, real sigma, real ter, row_vector stimulus_slots, row_vector stimulus_values,
      matrix result_slots, matrix result_values, row_vector activations){
      int idx;
      real LL[num_elements(activations)];
      int N;
      int j;
      N = num_elements(activations);
      // index of chunk mapping to yes response. This chunk could not have been retrieved for no response
      idx = get_chunk_index(result_slots, result_values, stimulus_slots, stimulus_values);
      j = 0;
      // compute the log likelihood of retrieving each possible chunk except idx
      for(i in 1:N){
        if(i != idx){
            j += 1;
            // log normal race log likelihood of response i at rt
            LL[j] = LNR_LL(-activations, sigma, ter, rt, i);
        }
      }
      return log_sum_exp(LL[1:j]);
  }

  real computeLL(real d, real blc, real delta, real gamma, int k, real L, int N, real sigma, real tau, real ter, int sa, int mp, int bll,
    int n_chunks, int n_slots, matrix memory_slots, matrix memory_values, row_vector stimulus_slots, row_vector stimulus_values, int resp,
    real rt){
    // real requestSlots[n_slots];
    // real requestValues[n_slots];
    row_vector[n_slots] imaginal_slots;
    row_vector[n_slots] imaginal_values;
    row_vector[n_chunks+1] activations; //n_chunks + threshold
    // slots: request_result[1,1]
    // values: request_result[1,2]
    // number of chunks: request_result[1,3][1,1]
    matrix[n_chunks,2] request_result[1,3];
    matrix[n_chunks,2] result_slots;
    matrix[n_chunks,2] result_values;
    real LLs[2];
    real lags[1];
    real denoms[n_slots];
    int n_results;

    lags[1] = 0;
    imaginal_slots = stimulus_slots;
    imaginal_values = stimulus_values;
    // pre-compute the denominator of fan values
    denoms = denom_spreading_activation(imaginal_values, memory_values, n_slots);

    if(resp==2){
      //Respond Yes

      // Return chunks that match retrieval request (stimulus_slots, stimulus_values). If mp = 1,
      // as in the current case, all chunk slots and values are returned
      request_result = request_chunks(stimulus_slots, stimulus_values, memory_slots, memory_values, mp);
      result_slots = request_result[1,1];
      result_values = request_result[1,2];
      n_results = to_int(request_result[1,3][1,1]);
      // compute activation for all chunks in the request result
      activations = compute_all_activations(blc, d, delta, gamma, tau, k, L, N, lags, sa, mp, bll, result_slots, result_values, stimulus_slots,
          stimulus_values, n_results, n_chunks, imaginal_slots, imaginal_values, denoms);
        // compute the log likelihood of a yes response
      return LLYes(rt, sigma, ter, stimulus_slots, stimulus_values, result_slots, result_values,
        activations[1:(n_results+1)]);
    }else{
      //Respond No

        // Return chunks that match retrieval request (stimulus_slots, stimulus_values). If mp = 1,
      // as in the current case, all chunk slots and values are returned
      request_result = request_chunks(stimulus_slots, stimulus_values, memory_slots, memory_values, mp);
      result_slots = request_result[1,1];
      result_values = request_result[1,2];
      n_results = to_int(request_result[1,3][1,1]);
      // compute activation for all chunks in the request result
      activations = compute_all_activations(blc, d, delta, gamma, tau, k, L, N, lags, sa, mp, bll, result_slots, result_values, stimulus_slots,
          stimulus_values, n_results, n_chunks, imaginal_slots, imaginal_values, denoms);
      // compute the log likelihood of a no response
      return LLNo(rt, sigma, ter, stimulus_slots, stimulus_values, result_slots, result_values,
        activations[1:(n_results+1)]);
    }
  }
}

data {
  int<lower=1> n_obs;
  int<lower=1> n_chunks;
  int<lower=1> n_slots;
  int<lower=0> resp[n_obs];
  real rts[n_obs];
  matrix[n_obs,2] stimuli_slots; //person,place
  matrix[n_obs,2] stimuli_values; //person,place
  // partial matching indicator
  int<lower=0,upper=1> mp;
  // base level learning indicator
  int<lower=0,upper=1> bll;
  // spreading activation indicator
  int<lower=0,upper=1> sa;
  // base level constant
  real blc;
  // retrieval threshold parameter
  real tau;
  // encoding and motor time 
  real ter;
  real s;
  matrix[n_chunks,2] memory_values; //person,place
  matrix[n_chunks,2] memory_slots; //person,place
}

parameters {
  real<lower=0> delta;
  real<lower=0> gamma;
}

model {
    real d;
    int k;
    real L;
    int N;
    real sigma;

    // Note that these variables are required by the functions are are not used in this model
    d = 0;
    k = 1;
    L = 0;
    N = 0;

    // standard deviation for activation noise
    sigma = s*pi()/sqrt(3);

    // prior distribution of mismatch penalty parameter
    delta ~ normal(.5, .25);
    // prior distribution of maximum association parameter
    gamma ~ normal(1.6, .8);

    for(trial in 1:n_obs){
      target += computeLL(d, blc, delta, gamma, k, L, N, sigma, tau, ter, sa, mp, bll, n_chunks, n_slots, memory_slots, memory_values,
         stimuli_slots[trial,:], stimuli_values[trial,:], resp[trial], rts[trial]);
    }
}
