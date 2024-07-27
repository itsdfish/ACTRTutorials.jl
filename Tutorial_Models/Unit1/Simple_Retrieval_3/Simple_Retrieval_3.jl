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
    return [Chunk(; act, value = i) for i = 1:n]
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
    actr = ACTR(; declarative = memory, parms..., τ, δ)
    data = map(x -> simulate_trial(actr, x), stimuli)
    return data
end

function simulate_trial(actr, stimulus)
    # Compute the retrieval probability of the chunk
    Θ, _ = retrieval_probs(actr; value = stimulus)
    n = length(Θ)
    idx = sample(1:n, Weights(Θ))
    # if idx corresponds to retrieval failure, guess between 1 and n-1
    idx = idx == n ? rand(1:(n - 1)) : idx
    # identify whether the response matches
    matches = idx == stimulus ? true : false
    return (matches = matches,)
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
    p_guess = 1 / n_items
    LL = 0.0
    LLs = zeros(typeof(τ), 2)
    for d in data
        if d.matches
            LLs[1] = log(p_correct)
            LLs[2] = log(p_failure) + log(p_guess)
        else
            LLs[1] = log(p_incorrect)
            LLs[2] = log(p_failure) + log(p_guess)
        end
        LL += logsumexp(LLs) * d.N
    end
    return LL
end
