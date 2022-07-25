using RCall

function expose(file)
    R"""
    library(rstan)
    expose_stan_functions($file)
    """
end


function stanLL(d, blc, delta, gamma, k, L, N, s, tau, sa, mp, bll, n_chunks, n_slots, memory_slots, memory_values, 
    stimulus_slots, stimulus_values, choice, choice_rep)
    LL = R"""
        computeLL($d, $blc, $delta, $gamma, $k, $L, $N, $s, $tau, $sa, $mp, $bll, $n_chunks, $n_slots, $memory_slots, $memory_values, 
            $stimulus_slots, $stimulus_values, $choice, $choice_rep)
    """
    return LL[1]
end
