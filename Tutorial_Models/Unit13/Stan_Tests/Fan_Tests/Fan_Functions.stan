//functions {

  real LLYes(real rt,real sigma,real ter,row_vector requestSlots,row_vector requestValues,matrix resultSlots,matrix resultValues,
    row_vector activations){
      int idx;
      idx = getChunkIndex(resultSlots,resultValues,requestSlots,requestValues);
      return LNR_LL(-activations,sigma,ter,rt,idx);
  }

  real LLNo(real rt, real sigma, real ter, row_vector stimulusSlots, row_vector stimulusValues,
      matrix resultSlots, matrix resultValues, row_vector activations){
      int idx;
      real LL[num_elements(activations)];
      int N;
      int j;
      N = num_elements(activations);
      idx = getChunkIndex(resultSlots, resultValues, stimulusSlots, stimulusValues);
      j = 0;
      for(i in 1:N){
        if(i != idx){
            j+=1;
            LL[j] = LNR_LL(-activations, sigma, ter, rt, i);
        }
      }
      return log_sum_exp(LL[1:j]);
  }

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

  real utility_match(row_vector chunkSlots, row_vector chunkValues, row_vector stimulusSlots,
     row_vector stimulusValues, real beta) {
       return mismatch(beta, chunkSlots, chunkValues, stimulusSlots,
    stimulusValues);
  }

  real utility_mismatch_person(row_vector chunkSlots, row_vector chunkValues, row_vector stimulusSlots,
     row_vector stimulusValues, real beta) {
      return matchPenalty(beta, [chunkSlots[2]], [chunkValues[2]], [stimulusSlots[2]],
    [stimulusValues[2]]);
  }

  real utility_mismatch_place(row_vector chunkSlots, row_vector chunkValues, row_vector stimulusSlots,
     row_vector stimulusValues, real beta) {
      return matchPenalty(beta, [chunkSlots[1]], [chunkValues[1]], [stimulusSlots[1]],
    [stimulusValues[1]]);
  }

  real utility_retrieval_failure(real beta) {
       return -beta;
  }

  row_vector get_utility(row_vector chunkSlots, row_vector chunkValues, row_vector stimulusSlots,
     row_vector stimulusValues, real beta) {
         row_vector[4] utils;
         utils[1] = utility_match(chunkSlots, chunkValues, stimulusSlots, stimulusValues, beta);
         utils[2] = utility_mismatch_person(chunkSlots, chunkValues, stimulusSlots, stimulusValues, beta);
         utils[3] = utility_mismatch_place(chunkSlots, chunkValues, stimulusSlots, stimulusValues, beta);
         utils[4] = utility_retrieval_failure(beta);
         return utils;
  }

  row_vector response_probs(row_vector chunkSlots, row_vector chunkValues, row_vector stimulusSlots,
     row_vector stimulusValues, real beta) {
         row_vector[4] utils;
         row_vector[4] probs;
         utils = get_utility(chunkSlots, chunkValues, stimulusSlots, stimulusValues, beta);
         probs = exp(utils)/sum(exp(utils));
         return probs;
  }

  real loglike_response(row_vector chunkSlots, row_vector chunkValues,
     row_vector stimulusSlots, row_vector stimulusValues, real beta, int resp) {
       row_vector[4] probs;
       real theta;
       probs = response_probs(chunkSlots, chunkValues, stimulusSlots, stimulusValues, beta);
       theta = probs[1];
       if(resp == 2){
         return log(theta);
       }else{
         return log(1-theta);
       }
  }

  real LLAll(real rt, int resp, real beta, real sigma, real ter, row_vector stimulusSlots, row_vector stimulusValues,
      matrix resultSlots, matrix resultValues, row_vector activations){
      real LL[num_elements(activations)];
      int N;
      N = num_elements(activations);
      for(i in 1:(N-1)){
          LL[i] = LNR_LL(-activations, sigma, ter, rt, i) +
          loglike_response(resultSlots[i,:], resultValues[i,:], stimulusSlots, stimulusValues, beta, resp);
      }
      LL[N] = LNR_LL(-activations, sigma, ter, rt, N) + loglike_response_failure(resp, beta);
      return log_sum_exp(LL);
  }
//}
