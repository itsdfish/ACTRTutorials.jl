//functions {


  real loglike_response_failure(int resp, real beta){
      row_vector[4] util;
      real prob;
      // responses: yes no no no
      util = [-beta,-beta,-beta,0.0];
      // probability responding yes given retrieval failure
      prob = exp(util[1])/sum(exp(util));
      // response is yes
      if(resp == 2){
          return log(prob);
      }else{
        return log(1-prob);
      }
  }

  real utility_match(row_vector chunk_slots, row_vector chunk_values, row_vector stimulus_slots,
     row_vector stimulus_values, real beta) {
       return mismatch(beta, chunk_slots, chunk_values, stimulus_slots,
    stimulus_values);
  }

  real utility_mismatch_person(row_vector chunk_slots, row_vector chunk_values, row_vector stimulus_slots,
     row_vector stimulus_values, real beta) {
      return match_penalty(beta, [chunk_slots[2]], [chunk_values[2]], [stimulus_slots[2]],
    [stimulus_values[2]]);
  }

  real utility_mismatch_place(row_vector chunk_slots, row_vector chunk_values, row_vector stimulus_slots,
     row_vector stimulus_values, real beta) {
      return match_penalty(beta, [chunk_slots[1]], [chunk_values[1]], [stimulus_slots[1]],
    [stimulus_values[1]]);
  }

  real utility_retrieval_failure(real beta) {
       return -beta;
  }

  row_vector get_utility(row_vector chunk_slots, row_vector chunk_values, row_vector stimulus_slots,
     row_vector stimulus_values, real beta) {
         row_vector[4] utils;
         utils[1] = utility_match(chunk_slots, chunk_values, stimulus_slots, stimulus_values, beta);
         utils[2] = utility_mismatch_person(chunk_slots, chunk_values, stimulus_slots, stimulus_values, beta);
         utils[3] = utility_mismatch_place(chunk_slots, chunk_values, stimulus_slots, stimulus_values, beta);
         utils[4] = utility_retrieval_failure(beta);
         return utils;
  }

  row_vector response_probs(row_vector chunk_slots, row_vector chunk_values, row_vector stimulus_slots,
     row_vector stimulus_values, real beta) {
         row_vector[4] utils;
         row_vector[4] probs;
         utils = get_utility(chunk_slots, chunk_values, stimulus_slots, stimulus_values, beta);
         probs = exp(utils)/sum(exp(utils));
         return probs;
  }

  real loglike_response(row_vector chunk_slots, row_vector chunk_values,
     row_vector stimulus_slots, row_vector stimulus_values, real beta, int resp) {
       row_vector[4] probs;
       real theta;
       probs = response_probs(chunk_slots, chunk_values, stimulus_slots, stimulus_values, beta);
       theta = probs[1];
       if(resp == 2){
         return log(theta);
       }else{
         return log(1-theta);
       }
  }

  real LLAll(real rt, int resp, real beta, real sigma, real ter, row_vector stimulus_slots, row_vector stimulus_values,
      matrix result_slots, matrix result_values, row_vector activations){
      real LL[num_elements(activations)];
      int N;
      N = num_elements(activations);
      for(i in 1:(N-1)){
          LL[i] = LNR_LL(-activations, sigma, ter, rt, i) +
          loglike_response(result_slots[i,:], result_values[i,:], stimulus_slots, stimulus_values, beta, resp);
      }
      LL[N] = LNR_LL(-activations, sigma, ter, rt, N) + loglike_response_failure(resp, beta);
      return log_sum_exp(LL);
  }
//}
