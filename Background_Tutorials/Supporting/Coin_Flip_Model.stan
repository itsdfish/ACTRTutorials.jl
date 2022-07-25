data {
  // number of trials
  int<lower=1> n;   
  // number of heads      
  int<lower=0> h;         
}

parameters {
  // probability of heads
  real<lower=0,upper=1> theta;      
}

model {
    // prior probability of heads
    theta ~ beta(5, 5);     
    // likelihood function   
    h ~ binomial(n, theta); 
}
