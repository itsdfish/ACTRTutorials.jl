using Distributions, StatsFuns
import Distributions: logpdf, loglikelihood

struct Retrieval{T1, T2, T3} <: ContinuousUnivariateDistribution
    τ::T1
    θg::T2
    fixed_parms::T3
end

Broadcast.broadcastable(x::Retrieval) = Ref(x)

loglikelihood(d::Retrieval, data::NamedTuple) = logpdf(d, data)

Retrieval(; τ, θg, fixed_parms) = Retrieval(τ, θg, fixed_parms)

function logpdf(d::Retrieval, data::NamedTuple)
    return computeLL(d.fixed_parms, data; τ = d.τ, θg = d.θg)
end

function simulate(parms, N; τ, θg)
    # create a chunk
    chunk = Chunk()
    # add the chunk to declarative memory
    memory = Declarative(; memory = [chunk])
    # create the ACTR object
    actr = ACTR(; declarative = memory, parms..., τ, θg)
    # compute the retrieval probability
    θᵣ, _ = retrieval_prob(actr, chunk)
    # compute the probabililty of a correct answer for target trials
    θ = θᵣ + (1 - θᵣ) * θg
    # generate responses for the target trial
    target = rand(Binomial(N.t, θ))
    # generate responses for foil trials
    foil = rand(Binomial(N.f, θg))
    return (Nt = N.t, Nf = N.f, target = target, foil = foil)
end

function computeLL(fixed_parms, data; τ, θg)
    # intitialize activation with the same variable type as τ to work with autodiff
    act = zero(typeof(τ))
    # initialize the chunk
    chunk = Chunk(; act)
    # add the chunk to declarative memory
    memory = Declarative(; memory = [chunk])
    # create the ACTR object
    actr = ACTR(; declarative = memory, fixed_parms..., τ, θg)
    # compute the retrival probability
    θᵣ, _ = retrieval_prob(actr, chunk)
    # compute the probability of a correct response
    θ = θᵣ + (1 - θᵣ) * θg
    # compute the log likelihood for target trials
    LL = logpdf(Binomial(data.Nt, θ), data.target)
    # compute the log likelihood for foil trials
    LL += logpdf(Binomial(data.Nf, θg), data.foil)
    return LL
end
