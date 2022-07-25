//functions {
real exactBaseLevel(real d, real[] lags){
  real act;
  act = 0.0;
  for (i in 1:size(lags)){
    act += pow(lags[i], -d);
  }
  return log(act);
}

real baseLevel(real d, int k, real L, int N, real[] lags){
  real exact;
  real approx;
  real tk;
  real x1;
  real x2;
  exact = exactBaseLevel(d, lags);
  approx = 0.0;
  if (N > k) {
    tk = lags[k];
    x1 = (N-k)*(pow(L,1-d)-tk^(1-d));
    x2 = (1-d)*(L-tk);
    approx = x1/x2;
  }
  return log(exp(exact) + approx);
}

int isIn(real requestSlot, row_vector chunkSlots){
  for(i in 1:num_elements(chunkSlots)){
    if(requestSlot == chunkSlots[i]){
        return 1;
    }
  }
  return 0;
}

int getSlotIndex(real requestSlot, row_vector chunkSlots){
  for(i in 1:num_elements(chunkSlots)){
    if(requestSlot == chunkSlots[i]){
        return i;
    }
  }
  return 0;
}

real mismatch(real delta, row_vector chunkSlots, row_vector chunkValues, row_vector requestSlots,
    row_vector requestValues){
  real p;
  int slotIndex;
  p = 0;
  for(i in 1:num_elements(requestSlots)){
    if(isIn(requestSlots[i], chunkSlots) == 0){
      p -= delta;
      continue;
    }
    else{
      slotIndex = getSlotIndex(requestSlots[i], chunkSlots);
      if(requestValues[i] != chunkValues[slotIndex]){
          p -= delta;
      }
    }
  }
  return p;
}

//Penalty when matches
real matchPenalty(real delta, row_vector chunkSlots, row_vector chunkValues, row_vector requestSlots,
    row_vector requestValues){
  real p;
  int slotIndex;
  p = 0;
  for(i in 1:num_elements(requestSlots)){
    if(isIn(requestSlots[i], chunkSlots) == 0){
      p -= delta;
      continue;
    }
    else{
      slotIndex = getSlotIndex(requestSlots[i], chunkSlots);
      if(requestValues[i] == chunkValues[slotIndex]){
          p -= delta;
      }
    }
  }
  return p;
}

int countValues(row_vector chunkValues, real value){
  int cnt;
  cnt = 0;
  for(i in 1:cols(chunkValues)){
    if (chunkValues[i] == value){
      cnt +=1;
    }
  }
  return cnt;
}

real computeDenom(matrix memoryValues, real imaginalValue){
  int N;
  int cnt;
  cnt = 0;
  N = rows(memoryValues);
  for(c in 1:N){
    cnt += countValues(memoryValues[c,:], imaginalValue);
  }
  return cnt;
}

real[] denomSpreadingActivation(row_vector imaginalValues, matrix memoryValues, int Nslots){
  real denoms[Nslots];
  for(i in 1:Nslots){
    denoms[i] = computeDenom(memoryValues, imaginalValues[i]);
  }
  return denoms;
}

real computeWeights(row_vector chunkValues){
  return 1.0/num_elements(chunkValues);
}

real spreadingActivation(row_vector chunkValues, row_vector imaginalValues, real[] denoms,
  real gamma){
    real w;
    real r;
    real sa;
    real fan;
    real num;
    r = 0;
    sa = 0;
    w = computeWeights(chunkValues);
    for(i in 1:num_elements(imaginalValues)){
      num = countValues(chunkValues, imaginalValues[i]);
      fan = num/(denoms[i] + 1);
      if(fan == 0){
        r = 0;
      }else{
        r = gamma + log(fan);
      }
      sa += w*r;
    }
    return sa;
  }

real computeActivation(real blc, real d, real delta, real gamma, int k, real L, int N, real[] lags,
  int sa, int mp, int bll, row_vector chunkSlots, row_vector chunkValues, row_vector requestSlots, row_vector requestValues,
  row_vector imaginalSlots, row_vector imaginalValues, real[] denoms){
    real act;
    act = blc;
    if(bll == 1){
      act += baseLevel(d, k, L, N, lags);
    }
    if(mp == 1){
      act += mismatch(delta, chunkSlots, chunkValues, requestSlots, requestValues);
    }
    if(sa ==1){
      act += spreadingActivation(chunkValues, imaginalValues, denoms, gamma);
    }
    return act;
}

real LNR_LL(row_vector mus, real sigma, real ter, real v, int c){
  real LL;
  LL = 0;
  for(i in 1:num_elements(mus)){
    if(i == c){
      LL += lognormal_lpdf(v-ter|mus[i],sigma);
  }else{
    LL += log(1-lognormal_cdf(v-ter,mus[i],sigma));
  }
}
  return LL;
}

real computeRetrievalProb(row_vector activations, real s, real tau, int N, int idx){
  real sigma;
  real prob;
  real v[N+1];
  sigma = s*sqrt(2);
  for(i in 1:N){
    v[i] = exp(activations[i]/sigma);
  }
  v[N+1] = exp(tau/sigma);
  prob = v[idx]/sum(v);
  return prob;
}

int allValuesMatch(row_vector requestSlots, row_vector chuckSlots,
  row_vector requestValues, row_vector chunkValues){
    int flag;
    for(s in 1:num_elements(requestSlots)){
      flag = 0;
      for(m in 1:num_elements(chuckSlots)){
        if(requestSlots[s] == chuckSlots[m] && requestValues[s] == chunkValues[m]){
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

int getChunkIndex(matrix resultSlots, matrix resultValues,
  row_vector requestSlots, row_vector requestValues){
    int flag;
    for(r in 1:rows(resultSlots)){
      flag = allValuesMatch(requestSlots, resultSlots[r,:], requestValues, resultValues[r,:]);
      if(flag == 1){
        return r;
      }
    }
    return -100;
}

matrix[,] requestChunks(row_vector requestSlots, row_vector requestValues,
  matrix memorySlots, matrix memoryValues, int mp){
    matrix[rows(memorySlots),cols(memorySlots)] matchSlots;
    matrix[rows(memorySlots),cols(memorySlots)] matchValues;
    matrix[rows(memorySlots),cols(memorySlots)] match[1,3];
    int Nrows;
    int chunkCount;
    int flag;
    //if mismatch penalty is active, return the memory matrix
    //No subsetting required
    if(mp == 1){
      match[1,1] = memorySlots;
      match[1,2] = memoryValues;
      match[1,3][1,1] = rows(memorySlots);
      return match;
    }

    chunkCount = 0;
    Nrows = rows(memoryValues);
    for(r in 1:Nrows){
      flag = allValuesMatch(requestSlots, memorySlots[r,:], requestValues, memoryValues[r,:]);
      if(flag == 1){
        chunkCount +=1;
        matchSlots[chunkCount,:] = memorySlots[r,:];
        matchValues[chunkCount,:] = memoryValues[r,:];
      }
    }
    match[1,1] = matchSlots;
    match[1,2] = matchValues;
    match[1,3][1,1] = chunkCount;
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

row_vector computeAllActivations(real blc, real d, real delta, real gamma, real tau, int k, real L, int N, real[] lags,
  int sa, int mp, int bll, matrix resultSlots, matrix resultValues, row_vector requestSlots,
    row_vector requestValues, int Nresults, int Nchunks, row_vector imaginalSlots, row_vector imaginalValues, real[] denoms){
    row_vector[Nchunks+1] activations;
    for(i in 1:Nresults){
      activations[i] = computeActivation(blc, d, delta, gamma, k, L, N, lags, sa, mp, bll, resultSlots[i,:],
        resultValues[i,:], requestSlots, requestValues, imaginalSlots, imaginalValues, denoms);
    }
    activations[Nresults+1] = tau;
    return activations;
  }

//}
