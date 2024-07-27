import Distributions: logpdf, loglikelihood

struct RT{T1, T2, T3} <: ContinuousUnivariateDistribution
    δ::T1
    τ::T2
    n_items::Int
    parms::T3
end

RT(; δ, τ, n_items, parms) = RT(δ, τ, n_items, parms)

loglikelihood(d::RT, data::Array{<:NamedTuple, 1}) = logpdf(d, data)

function logpdf(d::RT, data::Array{<:NamedTuple, 1})
    LL = computeLL(d.n_items, d.parms, data; δ = d.δ, τ = d.τ)
    return LL
end

function populate_memory(n, act = 0.0)
    return [Chunk(; act = act, value = i) for i = 1:n]
end

function sample_stimuli(n, reps)
    return rand(1:n, reps)
end

function simulate(n_items, stimuli, parms; δ, τ)
    # Create chunk
    chunks = populate_memory(n_items)
    # add chunk to declarative memory
    memory = Declarative(; memory = chunks)
    # create ACTR object and pass parameters
    actr = ACTR(; declarative = memory, parms..., δ, τ)
    n = length(stimuli)
    data = map(x -> simulate_trial(actr, x, n_items), stimuli)
    return data
end

function simulate_trial(actr, stimulus, n_items)
    # Retrieve chunk
    ter = get_parm(actr, :ter)
    # Retrieve a chunk
    chunk = retrieve(actr; value = stimulus)
    resp = 0
    # Code response index as 1 (correct), 2 (incorrect), n_items+1 (retrieval failure)
    if isempty(chunk)
        resp = n_items + 1
    else
        resp = chunk[1].slots.value == stimulus ? 1 : 2
    end
    # Compute reaction time 
    rt = compute_RT(actr, chunk) + ter
    return (resp = resp, stimulus = stimulus, rt = rt)
end

function computeLL(n_items, parms, data; δ, τ)
    (; ter, s) = parms
    LL = 0.0
    σ = s * pi / sqrt(3)
    act = zero(δ)
    chunks = populate_memory(n_items, act)
    # Add chunk and parameters to declarative memory
    memory = Declarative(; memory = chunks)
    # Create ACTR object
    actr = ACTR(; declarative = memory, parms..., δ, τ)
    actr.parms.noise = false
    # Compute activation given retrieval request (value,1)
    compute_activation!(actr; value = 1)
    # Get activation values
    Θs = map(x -> x.act, chunks)
    push!(Θs, τ)
    # Define  and cache distribution object
    dist = LNR(; μ = -Θs, σ, ϕ = ter)
    # Compute log likelihood for each data point
    for d in data
        # Compute log likelihood
        LL += logpdf(dist, d.resp, d.rt)
    end
    return LL
end
