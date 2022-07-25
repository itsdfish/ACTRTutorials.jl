using Parameters, Distributions, StatsFuns
import Distributions: logpdf

struct Retrieval{T1,T2,T3} <: ContinuousUnivariateDistribution
    τ::T1
    n_trials::T2
    parms::T3
end

Retrieval(;τ, n, parms) = Retrieval(τ, n, parms)

function logpdf(d::Retrieval, k::Int64)
    return computeLL(d.parms, d.n_trials, k; τ=d.τ)
end

function simulate(parms, n_trials; τ)
    # Create a chunk object
    chunk = Chunk()
    # Create a declarative memory object
    memory = Declarative(;memory=[chunk])
    # Create an ACTR object
    actr = ACTR(;declarative=memory, parms..., τ)
    # Compute the retrieval probability of the chunk
    θ,_ = retrieval_prob(actr, chunk)
    # Simulate n_trials
    data = rand(Binomial(n_trials, θ))
    return data
end

# primary function for computing log likelihood
function computeLL(parms, n_trials, k; τ)
    # initialize activation for auto-differentiation
    act = zero(typeof(τ))
    # create a chunk
    chunk = Chunk(;act=act)
    # add chunk and parameters to declarative memory
    memory = Declarative(;memory=[chunk])
    # add declarative memory to ACT-R object
    actr = ACTR(;declarative=memory, parms..., τ)
    # compute the probability of retrieving the chunk
    θᵣ,_ = retrieval_prob(actr, chunk)
    # compute the log likelihood of data given n_trials and θᵣ
    LL = logpdf(Binomial(n_trials, θᵣ), k)
    return LL
end
