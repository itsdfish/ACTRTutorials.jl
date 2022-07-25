import Distributions: logpdf, loglikelihood

struct RT{T1,T2,T3,T4} <: ContinuousUnivariateDistribution
    δ::T1
    s::T2
    τ::T3
    n_items::Int
    parms::T4
end

RT(;δ, s, τ, n_items, parms) = RT(δ, s, τ, n_items, parms)

loglikelihood(d::RT, data::Array{<:NamedTuple,1}) = logpdf(d, data)

function logpdf(d::RT, data::Array{<:NamedTuple,1})
    LL = computeLL(d.n_items, d.parms, data; δ = d.δ, s=d.s, τ=d.τ)
    return LL
end

function populate_memory(n, act=0.0)
    return [Chunk(;act=act, value=i) for i in 1:n]
end

function sample_stimuli(n, reps)
    return rand(1:n, reps)
end

function simulate(n_items, stimuli, parms; δ, s, τ)
    # Create chunk
    chunks = populate_memory(n_items)
    # add chunk to declarative memory
    memory = Declarative(;memory=chunks)
    # create ACTR object and pass parameters
    actr = ACTR(;declarative=memory, parms..., δ, s, τ)
    n = length(stimuli)
    data = map(x->simulate_trial(actr, n_items, x), stimuli)
    return data
end

function simulate_trial(actr, n_items, stimulus)
    # retrieve chunk
    ter = get_parm(actr, :ter)
    chunk = retrieve(actr; value = stimulus)
    rt,resp = 0.0,0
    if isempty(chunk)
        resp = rand(1:n_items)
        # compute reaction time 
        rt = compute_RT(actr, chunk) + ter + 0.05
    else
        resp = chunk[1].slots.value
        # compute reaction time 
        rt = compute_RT(actr, chunk) + ter
    end
    # correct: 1, incorrect: 2
    correct = resp == stimulus ? 1 : 2
    return (correct = correct, stimulus=stimulus, rt = rt)
end

function computeLL(n_items, parms, data; δ, s, τ)
    (;ter) = parms
    LLs = zeros(typeof(τ), 2)
    LL = 0.0
    σ = s * pi / sqrt(3)
    act = zero(δ)
    chunks = populate_memory(n_items, act)
    # add chunk to declarative memory
    memory = Declarative(;memory=chunks)
    # create ACTR object
    actr = ACTR(;declarative=memory, parms..., δ, s, τ) 
    actr.parms.noise = false
    # compute activation given retrieval request
    compute_activation!(actr; value=1)
    # get activation values
    Θs = map(x->x.act, chunks)
    push!(Θs, τ)
    # define distribution object for retrievals
    dist = LNR(;μ=-Θs, σ, ϕ=ter)
    # define distribution object for guessing after retrieval failure
    dist_guess = LNR(;μ=-Θs, σ, ϕ=(ter+0.05))
    # log probability of guess following retrieval failure
    lp_guess = log(1/n_items)
    # compute log likelihood for each data point
    for d in data
        # log likelihood of retrieving 
        LLs[1] = logpdf(dist, d.correct, d.rt)
        # log likelihood of guessing following retrieval failure
        LLs[2] = logpdf(dist_guess, n_items+1, d.rt) + lp_guess
        # increment log likelihiood of all data
        LL += logsumexp(LLs)
    end
    return LL
end