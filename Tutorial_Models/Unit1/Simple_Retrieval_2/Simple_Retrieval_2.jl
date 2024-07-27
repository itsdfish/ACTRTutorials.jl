using Parameters, Distributions, StatsFuns
import Distributions: logpdf, loglikelihood

struct Retrieval{T1, T2, T3, T4} <: ContinuousUnivariateDistribution
    τ::T1
    δ::T2
    n_items::T3
    parms::T4
end

loglikelihood(d::Retrieval, data::Array{<:NamedTuple, 1}) = logpdf(d, data)

function logpdf(d::Retrieval, data::Array{<:NamedTuple, 1})
    return computeLL(d.parms, d.n_items, data; τ = d.τ, δ = d.δ)
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
    # Add chunk and parameters to declarative memory
    memory = Declarative(; memory = chunks)
    # Create ACTR object
    actr = ACTR(; declarative = memory, parms..., τ, δ)
    # Generate data for each trial
    data = map(x -> simulate_trial(actr, x), stimuli)
    return data
end

function simulate_trial(actr, stimulus)
    # Compute the retrieval probability of the chunk
    Θ, _ = retrieval_probs(actr; value = stimulus)
    n = length(Θ)
    idx = sample(1:n, Weights(Θ))
    resp = :_
    # The last index corresponds to a retrieval failure
    if idx == n
        resp = :failure
        # Correct retrieval if match
    elseif idx == stimulus
        resp = :correct
        #Incorrect retrieval otherwise
    else
        resp = :incorrect
    end
    return (resp = resp,)
end

# primary function for computing log likelihood
function computeLL(parms, n_items, data; τ, δ)
    # initialize activation for auto-differentiation
    act = zero(τ)
    # populate declarative memory
    chunks = populate_memory(n_items, act)
    # add chunks to declarative memory
    memory = Declarative(; memory = chunks)
    # create ACTR object and pass parameters
    actr = ACTR(; declarative = memory, parms..., τ, δ)
    # pre-compute probabilities for correct, incorrect and failures
    p_correct, p_failure = retrieval_prob(actr, chunks[1]; value = 1)
    p_incorrect, _ = retrieval_prob(actr, chunks[1]; value = 2)
    # initialize log likelihood
    LL = 0.0
    # for each unique data, compute loglikelihood and multiply by counts
    for d in data
        if d.resp == :correct
            LL += log(p_correct) * d.N
        elseif d.resp == :incorrect
            LL += log(p_incorrect) * d.N
        else
            LL += log(p_failure) * d.N
        end
    end
    return LL
end
