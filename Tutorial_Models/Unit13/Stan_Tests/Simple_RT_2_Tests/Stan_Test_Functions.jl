using RCall

function expose(file)
    R"""
    library(rstan)
    expose_stan_functions($file)
    """
end

function stanLL(blc, tau, sigma, ter, resp, rt)
    LL = R"""
        computeLL($blc, $tau, $sigma, $ter, $resp, $rt)
    """
    return LL[1]
end
