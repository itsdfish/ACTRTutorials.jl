function prob_nonattentive_lapse(υ, τ, λ, s, n_lapses)
    v1 = exp((υ*(λ^n_lapses))./s)
    v2 = exp(τ/s)
    return v2/(1 + v1 + v2)
end

function prob_attentive_lapse(υ, τ, λ, s, n_lapses)
    v1 = exp((υ*(λ^n_lapses))./s)
    v2 = exp(τ/s)
    return v2/(v1 + v2)
end

function prob_attentive_resp(υ, τ, λ, s, n_lapses)
    v1 = exp((υ*(λ^n_lapses))/s)
    v2 = exp(τ/s)
    return v1/(v1 + v2)
end

function prob_attend(υ, τ, λ, s, n_lapses)
    v1 = exp((υ*(λ^n_lapses))/s)
    v2 = exp(τ/s)
    return v1/(1 + v1 + v2)
end

function prob_nonattentive_resp(υ, τ, λ, s, n_lapses)
    v1 = exp((υ*(λ^n_lapses))/s)
    v2 = exp(τ/s)
    return 1/(1 + v1 + v2)
end

function nonattentive_mixture(υ, τ, λ, s, n_cycles)
    weight = 1.0
    # compute joint probability of n_cycles - 1 lapses 
    for n_lapses in 0:(n_cycles-2)
        weight *= prob_nonattentive_lapse(υ, τ, λ, s, n_lapses)
    end
    # multiply weight by probability of non-attentive response
    return weight *= prob_nonattentive_resp(υ, τ, λ, s, n_cycles-1)
end

function attentive_mixture(υ, τ, λ, s, n_cycles)
    weight = 0.0
    # loop through all of the ways of attending and respond with n_cycles - 2 lapses
    # e.g. (A,L,L,R), (L,A,L,R), (L,L,A,R) for n_cycles = 3
    for n_lapses in 0:(n_cycles-2)
        lapse_weight = 1.0
        # joint probability of n_lapses prior to attending
        for i in 0:(n_lapses-1)
            lapse_weight *= prob_nonattentive_lapse(υ, τ, λ, s, i)
        end
        # multiply lapse_weight by attend probabilty
        lapse_weight *= prob_attend(υ, τ, λ, s, n_lapses)
        # multiply lapse_weight by the joint probability of lapses after attending
        for i in n_lapses:(n_cycles-3)
            lapse_weight *= prob_attentive_lapse(υ, τ, λ, s, i)
        end
        # add sequence to weight variable
        weight += lapse_weight
    end
    # multiply weight by probability of attentive response
    return weight *= prob_attentive_resp(υ, τ, λ, s, n_cycles-2)
end

function post_stimulus_prob(υ, τ, λ, s, n_cycles)
    # non attentive respose mixture probability
    nonatt_resp = nonattentive_mixture(υ, τ, λ, s, n_cycles)
    # attentive Response mixture probability
    att_resp = attentive_mixture(υ, τ, λ, s, n_cycles)
    return [nonatt_resp att_resp]
end


function mixture_parms(max_cycles; γ, mt=.06, ρ=2/3)
	mu = fill(0.0, max_cycles, 2)
	sigma = similar(mu)
	cycles = 1:max_cycles
	# non-attentive process
	@. mu[:,1] = cycles * γ + mt 
	# attentive process
	@. mu[:,2] = cycles * γ  + .085 + mt 
	# non-attentive process
	@. sigma[:,1] = 1.05 * sqrt(
		(1/12) * (cycles * (γ * ρ).^2 + (mt * ρ)^2)) 
	# attentive process
	@. sigma[:,2] = 1.05*sqrt.(
	(1/12) * (cycles * (γ * ρ)^2  + (.085 * ρ)^2
		+ (mt * ρ)^2))
	return mu,sigma
end

function compute_mixture_weights(υ, τ, λ, s, max_cycles)
    probs = zeros(typeof(υ), max_cycles, 2)
    for n_cycles in 1:max_cycles
        probs[n_cycles,:] = post_stimulus_prob(υ, τ, λ, s, n_cycles)
    end
    return probs
end

"""
    pvt_log_like(rt::Float64; υ, τ, λ, γ, s=.45345, max_cycles=15)

Computes the log likelihood of a vector of rts for the PVT model. Note that
this does not handle false starts. The parameters are as follows:
-`υ`: utility
-`τ`: threshold
-`λ`: FPdec
-`γ`: conflict resolution time
-`rts`: the data
-`s`: utility noise default
-`max_cycles`: maximum number of component distributions,  each corresponding
    to a latent number of production cycles. 15 is sufficient for most parameterizations.
"""
pvt_log_like(rt::Float64; υ, τ, λ, γ, s=.45345, max_cycles=15) = pvt_log_like([rt]; υ, τ, λ, γ, s, max_cycles)

function pvt_log_like(RTs::Vector{Float64}; υ, τ, λ, γ, s=.45345, max_cycles=15)
    LL = 0.0
    # compute parameters of Normal distribution for for production cycles up to max_cycles
    mus, sigmas = mixture_parms(max_cycles; γ)
    # compute weights for each mixture distribution for for production cycles up to max_cycles
    weights = compute_mixture_weights(υ, τ, λ, s, max_cycles)
    for rt in RTs
        # compute marginal likelihood of rt
        L = likelihood_trial(rt, mus, sigmas, weights, max_cycles)
        LL += log(L)
    end
    return LL
end

function likelihood_trial(rt, mus, sigmas, weights, max_cycles)
    L = 0.0
    # loop over non-attend and attend states
    for att in 1:2
        # loop over each number of production cycles
        for n_cycles in 1:max_cycles
            # add mixture density to likelihood L
            L += weights[n_cycles, att] * pdf(
                Normal(mus[n_cycles, att], 
                sigmas[n_cycles, att]
            ), rt)
        end
    end
    return L
end

simulate(;υ, τ, λ, γ, n_trials) = simulate(υ, τ, λ, γ, n_trials)

function simulate(υ, τ, λ, γ, n_trials)
    rts = zeros(n_trials)
    for trial in 1:n_trials
        rts[trial] = simulate_trial(υ, τ, λ, γ)
    end
    return rts
end

function simulate_trial(υ, τ, λ, γ)
    s = 0.45345     # utility noise
    a_time = 0.085  # encoding time
    r_time = 0.06   # response time
    ub = 10 		# time limit in seconds
    state = 1 		# [1 = attend 2 = respond]
    t = 0.0 		# model run time
    n_ml = 0 		# microlapse count
    while (state == 1) && (t < ub) ##signal present
        utility =  υ*λ^n_ml*[1.0,  0.0] .+ rand(Logistic(0.0, s), 2)
        t += rand_time(γ)  ## duration of conflict resolution
        max_util,max_idx = findmax(utility) ##conflict resolution
        if max_util < τ ##no production exceeds threshold
            n_ml += 1 ## increase microlapse count
        elseif max_idx == 1 ## attend utility exceeds threshold
            state = 2 ##prepare to respond
            t += rand_time(a_time)
        elseif max_idx == 2 ## respond utility exceeds threshold
            state = 3 ##respond
            t += rand_time(r_time)
        end
    end

    while (state == 2) && (t < ub) ## encoding complete
        utility = υ*λ^n_ml + rand(Logistic(0.0, s))
        t += rand_time(γ) ## duration of conflict resolution
        if utility < τ ## no production exceeds threshold
            n_ml += 1 ## increase microlapse count
        else ## respond utility exceeds threshold
            state = 3 ##respond
            t += rand_time(r_time)
        end
    end
    return t
end

rand_time(μ) = rand(Uniform(2 / 3 * μ, 4 / 3 * μ))