//functions {
real exact_base_level(real d, real[] lags){
  real act;
  act = 0.0;
  for (i in 1:size(lags)){
    act += pow(lags[i], -d);
  }
  return log(act);
}

real base_level(real d, int k, real L, int N, real[] lags){
  real exact;
  real approx;
  real tk;
  real x1;
  real x2;
  exact = exact_base_level(d, lags);
  approx = 0.0;
  if (N > k) {
    tk = lags[k];
    x1 = (N-k)*(pow(L,1-d)-tk^(1-d));
    x2 = (1-d)*(L-tk);
    approx = x1/x2;
  }
  return log(exp(exact) + approx);
}

int isIn(real request_slot, row_vector chunk_slots){
  for(i in 1:num_elements(chunk_slots)){
    if(request_slot == chunk_slots[i]){
        return 1;
    }
  }
  return 0;
}

int get_slot_index(real request_slot, row_vector chunk_slots){
  for(i in 1:num_elements(chunk_slots)){
    if(request_slot == chunk_slots[i]){
        return i;
    }
  }
  return 0;
}

real mismatch(real delta, row_vector chunk_slots, row_vector chunk_values, row_vector request_slots,
    row_vector request_values){
  real p;
  int slot_index;
  p = 0;
  for(i in 1:num_elements(request_slots)){
    if(isIn(request_slots[i], chunk_slots) == 0){
      p -= delta;
      continue;
    }
    else{
      slot_index = get_slot_index(request_slots[i], chunk_slots);
      if(request_values[i] != chunk_values[slot_index]){
          p -= delta;
      }
    }
  }
  return p;
}

//Penalty when matches
real match_penalty(real delta, row_vector chunk_slots, row_vector chunk_values, row_vector request_slots,
    row_vector request_values){
  real p;
  int slot_index;
  p = 0;
  for(i in 1:num_elements(request_slots)){
    if(isIn(request_slots[i], chunk_slots) == 0){
      p -= delta;
      continue;
    }
    else{
      slot_index = get_slot_index(request_slots[i], chunk_slots);
      if(request_values[i] == chunk_values[slot_index]){
          p -= delta;
      }
    }
  }
  return p;
}

int count_values(row_vector chunk_values, real value){
  int cnt;
  cnt = 0;
  for(i in 1:cols(chunk_values)){
    if (chunk_values[i] == value){
      cnt +=1;
    }
  }
  return cnt;
}

real compute_denom(matrix memory_values, real imaginal_value){
  int N;
  int cnt;
  cnt = 0;
  N = rows(memory_values);
  for(c in 1:N){
    cnt += count_values(memory_values[c,:], imaginal_value);
  }
  return cnt;
}

real[] denom_spreading_activation(row_vector imaginal_values, matrix memory_values, int Nslots){
  real denoms[Nslots];
  for(i in 1:Nslots){
    denoms[i] = compute_denom(memory_values, imaginal_values[i]);
  }
  return denoms;
}

real compute_weights(row_vector chunk_values){
  return 1.0 / num_elements(chunk_values);
}

real spreading_activation(row_vector chunk_values, row_vector imaginal_values, real[] denoms,
  real gamma){
    real w;
    real r;
    real sa;
    real fan;
    real num;
    r = 0;
    sa = 0;
    w = compute_weights(chunk_values);
    for(i in 1:num_elements(imaginal_values)){
      num = count_values(chunk_values, imaginal_values[i]);
      fan = num/(denoms[i] + 1);
      if(fan == 0){
        r = 0;
      }else{
        r = gamma + log(fan);
      }
      sa += w * r;
    }
    return sa;
  }

real compute_activation(real blc, real d, real delta, real gamma, int k, real L, int N, real[] lags,
  int sa, int mp, int bll, row_vector chunk_slots, row_vector chunk_values, row_vector request_slots, row_vector request_values,
  row_vector imaginal_slots, row_vector imaginal_values, real[] denoms){
    real act;
    act = blc;
    if(bll == 1){
      act += base_level(d, k, L, N, lags);
    }
    if(mp == 1){
      act += mismatch(delta, chunk_slots, chunk_values, request_slots, request_values);
    }
    if(sa ==1){
      act += spreading_activation(chunk_values, imaginal_values, denoms, gamma);
    }
    return act;
}

real LNR_LL(row_vector mus, real sigma, real ter, real v, int c){
  real LL;
  LL = 0;
  for(i in 1:num_elements(mus)){
    if(i == c){
      LL += lognormal_lpdf(v - ter|mus[i], sigma);
  }else{
    LL += log(1  -lognormal_cdf(v - ter, mus[i], sigma));
  }
}
  return LL;
}

real compute_retrieval_prob(row_vector activations, real s, real tau, int n_chunks, int idx){
  real sigma;
  real prob;
  real v[n_chunks+1];
  sigma = s * sqrt(2);
  for(i in 1:n_chunks){
    v[i] = exp(activations[i] / sigma);
  }
  v[n_chunks+1] = exp(tau / sigma);
  prob = v[idx] / sum(v);
  return prob;
}

int all_values_match(row_vector request_slots, row_vector chuck_slots,
  row_vector request_values, row_vector chunk_values){
    int flag;
    for(s in 1:num_elements(request_slots)){
      flag = 0;
      for(m in 1:num_elements(chuck_slots)){
        if(request_slots[s] == chuck_slots[m] && request_values[s] == chunk_values[m]){
          flag = 1;
          break;
        }
      }
      if(flag == 0){
        return 0;
      }
    }
    return 1;
}

real marginal_retrieval_prob(row_vector activations, real s, real tau, matrix result_slots, matrix result_values,
  row_vector response_slots, row_vector response_values, int n_results) {
    int flag;
    real prob;
    int n_chunks;
    prob = 0;
    //print("Choice slot ", response_slots[1], " Choice value ", response_values[1]);
    for(r in 1:n_results){
      flag = all_values_match(response_slots, result_slots[r,:], response_values, result_values[r,:]);
      if(flag == 1){
       // print("chunk index ", r, " prob ", compute_retrieval_prob(activations, s, tau, n_results, r));
        prob += compute_retrieval_prob(activations, s, tau, n_results, r);
      }
    }
    return prob;
}

int get_chunk_index(matrix result_slots, matrix result_values,
  row_vector request_slots, row_vector request_values){
    int flag;
    for(r in 1:rows(result_slots)){
      flag = all_values_match(request_slots, result_slots[r,:], request_values, result_values[r,:]);
      if(flag == 1){
        return r;
      }
    }
    return -100;
}

matrix[,] request_chunks(row_vector request_slots, row_vector request_values,
  matrix memory_slots, matrix memory_values, int mp){
    matrix[rows(memory_slots),cols(memory_slots)] match_slots;
    matrix[rows(memory_slots),cols(memory_slots)] match_values;
    matrix[rows(memory_slots),cols(memory_slots)] match[1,3];
    int n_rows;
    int chunk_count;
    int flag;
    //if mismatch penalty is active, return the memory matrix
    //No subsetting required
    if(mp == 1){
      match[1,1] = memory_slots;
      match[1,2] = memory_values;
      match[1,3][1,1] = rows(memory_slots);
      return match;
    }

    chunk_count = 0;
    n_rows = rows(memory_values);
    for(r in 1:n_rows){
      flag = all_values_match(request_slots, memory_slots[r,:], request_values, memory_values[r,:]);
      if(flag == 1){
        chunk_count += 1;
        match_slots[chunk_count,:] = memory_slots[r,:];
        match_values[chunk_count,:] = memory_values[r,:];
      }
    }
    match[1,1] = match_slots;
    match[1,2] = match_values;
    match[1,3][1,1] = chunk_count;
    return match;
}

int to_int(real v){
  int i;
  i = 1;
  while (i < v){
    i+=1;
  }
  return i;
}

row_vector compute_all_activations(real blc, real d, real delta, real gamma, real tau, int k, real L, int N, real[] lags,
  int sa, int mp, int bll, matrix result_slots, matrix result_values, row_vector request_slots,
    row_vector request_values, int n_results, int n_chunks, row_vector imaginal_slots, row_vector imaginal_values, real[] denoms){
    row_vector[n_chunks+1] activations;
    //row_vector[n_results+1] activations;
    for(i in 1:n_results){
      activations[i] = compute_activation(blc, d, delta, gamma, k, L, N, lags, sa, mp, bll, result_slots[i,:],
        result_values[i,:], request_slots, request_values, imaginal_slots, imaginal_values, denoms);
    }
    activations[n_results+1] = tau;
    return activations;
  }

//}
