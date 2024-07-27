using RCall

function expose(file)
    R"""
    library(rstan)
    expose_stan_functions($file)
    """
end

function stanLL(d, blc, delta, gamma, k, L, N, sigma, tau, ter, sa, mp, bll, Nchunks,
    Nslots, memorySlots, memoryValues,
    sslots, svalues, response, rt)
    LL = R"""
    computeLL($d, $blc, $delta, $gamma, $k, $L, $N, $sigma, $tau, $ter, $sa, $mp, $bll, $Nchunks, $Nslots, $memorySlots, $memoryValues,
    $sslots, $svalues, $response, $rt)
    """
    return LL[1]
end
