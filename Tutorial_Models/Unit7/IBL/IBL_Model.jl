using Parameters, Distributions, StatsFuns, StatsBase
import Distributions: logpdf, loglikelihood
import StatsBase: sample

struct IBL{T1, T2, T3, T4} <: ContinuousUnivariateDistribution
    d::T1
    ϕ::T2
    parms::T3
    gambles::T4
end

Broadcast.broadcastable(x::IBL) = Ref(x)

IBL(; d, ϕ, parms, gambles) = IBL(d, ϕ, parms, gambles)

loglikelihood(d::IBL, data::Array{<:T, 1}) where {T <: Array{<:NamedTuple, 1}} =
    logpdf(d, data)

function logpdf(dist::IBL, data::Array{<:T, 1}) where {T <: Array{<:NamedTuple, 1}}
    return computeLL(dist.parms, dist.gambles, data; d = dist.d, ϕ = dist.ϕ)
end

mutable struct Gamble{T1, T2}
    v::T1
    p::T2
end

Gamble(v::T) where {T <: Real} = Gamble([v], [1.0])

function Gamble(; v, p)
    return Gamble(v, p)
end

function Gambles()
    temp = (a = Gamble([2.0, -1.0], [0.5, 0.5]), b = Gamble(0.0))
    gambles = Array{typeof(temp), 1}()
    push!(gambles, temp)
    temp = (a = Gamble([3.0, 1.0], [0.8, 0.2]), b = Gamble([4.0, 2.0], [0.5, 0.5]))
    push!(gambles, temp)
    temp = (a = Gamble([-10.0, -0.0], [0.5, 0.5]), b = Gamble(-4.0))
    push!(gambles, temp)
    return gambles
end

sample(G::NamedTuple, option::Symbol) = sample(getfield(G, option))

function sample(g::Gamble)
    w = Weights(g.p)
    return sample(g.v, w)
end

EV(g) = g.v' * g.p

function populate_memory(gamble, act = 0.0)
    chunks = [Chunk(; act = act, choice = k, outcome = 30.0) for k in keys(gamble)]
    return chunks
end

function simulate(parms, gambles, n_trials; d, ϕ)
    # initialize one chunk for each option
    chunks = populate_memory(gambles)
    # populate declarative memroy
    memory = Declarative(; memory = chunks)
    # create the ACT-R object and add parameters
    actr = ACTR(; declarative = memory, parms..., d, ϕ)
    # extract the gamble names
    choices = [keys(gambles)...]
    data = Array{NamedTuple, 1}(undef, n_trials)
    for (i, trial) in enumerate(1.0:n_trials)
        # return a vector representing the probability of choosing each option
        θ = decision_probs(actr, gambles, trial)
        # sample a choice based on choice probabilities θ
        choice = sample(choices, Weights(θ))
        # sample an outcome from the choosen option
        outcome = sample(gambles, choice)
        # add trial, choice, and outcome information to data array
        data[i] = (trial = trial, choice = choice, outcome = outcome)
        # adds new chunk, or increments N and time stamp for existing chunk
        add_chunk!(actr, trial; choice, outcome)
    end
    return [data...]
end

function compute_utility(actr, trial, choice)
    # return retrieval probability p and chunks given request choice = choice
    p, chunks = retrieval_probs(actr, trial; choice)
    # remove the last value, which represents a negligible probability of a retrieval failure
    pop!(p)
    # extract the utilities from the chunks
    u = map(x -> x.slots.outcome, chunks)
    # compute the expected utility
    return p' * u
end

function get_index(choices, choice)
    for (i, v) in enumerate(choices)
        v == choice ? (return i) : nothing
    end
    return 0
end

function decision_prob(actr, gambles, choice, trial)
    p = decision_probs(actr, gambles, trial)
    choices = [keys(gambles)...]
    idx = get_index(choices, choice)
    return p[idx]
end

function decision_probs(actr, gambles, trial)
    ϕ = actr.parms.misc.ϕ
    choices = [keys(gambles)...]
    f(c) = compute_utility(actr, trial, c)
    u = map(f, choices)
    p = exp.(u / ϕ) ./ sum(exp.(u / ϕ))
    return p
end

function computeLL(parms, gamble_set, Data; d, ϕ)
    # initialize the log likelihood
    LL = zero(d)
    # compute log likelihood for all trials in a given block. Each block corresponds to a gamble
    for (data, gambles) in zip(Data, gamble_set)
        LL += computeLLBlock(parms, gambles, data; d, ϕ)
    end
    return LL
end

function computeLLBlock(parms, gambles, data; d, ϕ)
    T = typeof(d)
    act = zero(d)
    # initialize a chunk for each option in gambles
    chunks = populate_memory(gambles, act)
    # populate declarative memory
    memory = Declarative(; memory = chunks)
    # create an ACT-R object and pass parameters 
    actr = ACTR(; declarative = memory, parms..., d, ϕ)
    LL::T = 0.0
    for v in data
        # compute the decision probability
        p = decision_prob(actr, gambles, v.choice, v.trial)
        # increment the log likelihood
        LL += log(p)
        # update declarative memory based on choice and outcome
        add_chunk!(actr, v.trial; bl = act, choice = v.choice, outcome = v.outcome)
    end
    return LL
end

function posterior_a_rate(parms, gambles, n_trials; d, ϕ)
    data = map(x -> simulate(parms, x, n_trials; d, ϕ), gambles)
    return a_rate.(data)
end

function a_rate(data)
    a = 0
    N = length(data) - 1
    for i = 1:N
        a += data[i].choice != data[i + 1].choice ? 1 : 0
    end
    return a / N
end

function recurrence_rate(x)
    cnt = 0
    n = length(x)
    for i = 1:n
        for j = 1:n
            cnt += i ≠ j ? x[i] == x[j] : 0
        end
    end
    return cnt / (n^2 - n)
end

recurrence_indices(x::Vector{<:NamedTuple}) = recurrence_indices(map(x -> x.choice, x))

function recurrence_indices(x)
    cnt = 0
    n = length(x)
    v1 = Int[]
    v2 = Int[]
    for i = 1:n
        for j = 1:n
            if (i ≠ j) && (x[i] == x[j])
                push!(v1, i)
                push!(v2, j)
            end
        end
    end
    return v1, v2
end
