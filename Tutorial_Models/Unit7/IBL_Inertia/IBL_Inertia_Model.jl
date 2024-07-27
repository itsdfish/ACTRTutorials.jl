using Parameters, Distributions, StatsFuns, StatsBase
import Distributions: logpdf, loglikelihood
import StatsBase: sample

struct IBL{T1, T2, T3, T4, T5} <: ContinuousUnivariateDistribution
    d::T1
    ϕ::T2
    ρ::T3
    parms::T4
    gambles::T5
end

Broadcast.broadcastable(x::IBL) = Ref(x)

IBL(; d, ϕ, ρ, parms, gambles) = IBL(d, ϕ, ρ, parms, gambles)

loglikelihood(d::IBL, data::Array{<:T, 1}) where {T <: Array{<:NamedTuple, 1}} =
    logpdf(d, data)

function logpdf(dist::IBL, data::Array{T, 1}) where {T <: Array{<:NamedTuple, 1}}
    return computeLL(dist.parms, dist.gambles, data; d = dist.d, ϕ = dist.ϕ, ρ = dist.ρ)
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

function simulate(parms, gambles, n_trials; d, ϕ, ρ)
    # initialize a chunk for each gamble
    chunks = populate_memory(gambles)
    # populate declarative memory
    memory = Declarative(; memory = chunks)
    # create ACT-R object and pass parameters
    actr = ACTR(; declarative = memory, parms..., d, ϕ, ρ)
    choices = [keys(gambles)...]
    data = Array{NamedTuple, 1}(undef, n_trials)
    # initialize blank variable for previous option
    prev = :_
    for (i, trial) in enumerate(1.0:n_trials)
        # select a response
        choice = decide(actr, choices, gambles, trial, prev)
        # sample an outcome from the choosen option
        outcome = sample(gambles, choice)
        # add trial, choice, and outcome information to data array
        data[i] = (trial = trial, choice = choice, outcome = outcome, prev = prev)
        # adds new chunk or updates N for existing chunk
        add_chunk!(actr, trial; choice, outcome)
        # update previous choice
        prev = choice
    end
    return [data...]
end

function decide(actr, choices, gamble, trial, prev)
    ρ = get_parm(actr, :ρ)
    choice = :_
    if prev != :_ && rand() <= ρ
        choice = prev
    else
        θ = decision_probs(actr, gamble, trial)
        choice = sample(choices, Weights(θ))
    end
    return choice
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

function computeLL(parms, gambles, Data; d, ϕ, ρ)
    # initialize the log likelihood
    LL = 0.0
    # compute log likelihood for all trials in a given block. Each block corresponds to a gamble
    for (data, gamble) in zip(Data, gambles)
        LL += computeLLBlock(parms, gamble, data; d, ϕ, ρ)
    end
    return LL
end

function computeLLBlock(parms, gambles, data; d, ϕ, ρ)
    T = typeof(d)
    act = zero(T)
    # initialize a chunk for each option in gambles
    chunks = populate_memory(gambles, act)
    # populate declarative memory
    memory = Declarative(; memory = chunks)
    # create an ACT-R object and pass parameters 
    actr = ACTR(; declarative = memory, parms..., d, ϕ, ρ)
    LL::T = 0.0
    for v in data
        # compute the decision probability
        LL += decisionLL(actr, v, gambles; d, ϕ, ρ)
        # update declarative memory based on choice and outcome
        add_chunk!(actr, v.trial; bl = act, choice = v.choice, outcome = v.outcome)
    end
    return LL
end

function decisionLL(actr, data, gambles; d, ϕ, ρ)
    @unpack prev, choice, trial = data
    # compute the decision probability
    pr = decision_prob(actr, gambles, choice, trial)
    if trial == 1
        # mixture when previous and current choices are equal
    elseif prev == choice
        pr = (1 - ρ) * pr + ρ
        # mixture when previous and current choices are different    
    else
        pr = (1 - ρ) * pr
    end
    return log(pr)
end

function posterior_a_rate(parms, gambles, n_samples; d, ϕ, ρ)
    data = map(x -> simulate(parms, x, n_trials; d, ϕ, ρ), gambles)
    return a_rate.(data)
end

function a_rate(data)
    a = 0
    N = length(data) - 1
    for i = 1:N
        if data[i].choice != data[i + 1].choice
            a += 1
        end
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

function recurrence_indices(x1::Vector{<:NamedTuple}, x2::Vector{<:NamedTuple})
    recurrence_indices(map(x -> x.choice, x1), map(x -> x.choice, x2))
end

function recurrence_indices(x1, x2)
    cnt = 0
    n = length(x)
    v1 = Int[]
    v2 = Int[]
    for i = 1:n
        for j = 1:n
            if x1[i] == x2[j]
                push!(v1, i)
                push!(v2, j)
            end
        end
    end
    return v1, v2
end
