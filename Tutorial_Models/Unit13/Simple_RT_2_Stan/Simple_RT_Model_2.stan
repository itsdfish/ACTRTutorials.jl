functions {
    #include ../Stan_Utilities/ACTR_Functions.stan

    real computeLL(real blc, real tau, real sigma, real ter, int resp, real rt){
      // log normal race log likelihood called from ../Stan_Utilities/ACTR_Functions.stan
      return LNR_LL([-blc, -tau], sigma, ter, rt, resp);
    }
}

data {
  int<lower=1> n_obs;         // number of observations
  int<lower=0> resp[n_obs];   // response indices
  real rts[n_obs];            // rts
  real ter;                   // fixed ter paramter
  real s;                     // fixed logistic scalar parameter
}
parameters {
  real blc;                   // base level constant
  real tau;                   // retrieval threshold
}

model {
    real sigma;                    // activation noise parameter
    sigma = s * pi() / sqrt(3);    

    blc ~ normal(1.25, 0.5);       // prior for base level constant
    tau ~ normal(.5, .5);          // prior for retrieval threshold

    // loop over data and compute log likelihood of each response
    for(trial in 1:n_obs){
      target += computeLL(blc, tau, sigma, ter, resp[trial], rts[trial]);
    }
}
