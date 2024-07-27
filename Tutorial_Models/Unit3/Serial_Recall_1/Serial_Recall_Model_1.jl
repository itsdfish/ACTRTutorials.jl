using Parameters, Distributions, StatsFuns, StatsBase
import Distributions: logpdf, rand, loglikelihood

struct SerialRecall{T1, T2} <: ContinuousUnivariateDistribution
    δ::T1
    parms::T2
    n_trials::Int
end

Broadcast.broadcastable(x::SerialRecall) = Ref(x)

SerialRecall(; δ, parms, n_trials) = SerialRecall(δ, parms)

loglikelihood(d::SerialRecall, data::Array{<:Array{<:NamedTuple, 1}, 1}) = logpdf(d, data)

function logpdf(d::SerialRecall, data::Array{<:Array{<:NamedTuple, 1}, 1})
    return computeLL(d.parms, data, d.n_trials; δ = d.δ)
end

function populate_memory(n_items, act = 0.0)
    chunks = [Chunk(; act = act, position = i, retrieved = [false]) for i = 1:n_items]
    return chunks
end

# function simulate(parms,Nitems=10; δ)
#     chunks = populate_memory(Nitems)
#     memory = Declarative(;memory=chunks, parms..., δ=δ)
#     actr = ACTR(;declarative=memory)
#     data = Array{NamedTuple,1}(undef,Nitems)
#     resp=0
#     for i in 1:Nitems
#         chunk = retrieve(actr;position=i)
#         isempty(chunk) ? resp=-100 : resp = chunk[1].slots.position
#         data[i] = (position=i,resp=resp)
#     end
#     return data
# end

function simulate(parms, n_items = 10; δ)
    chunks = populate_memory(n_items)
    # populate declarative memory with chunks
    memory = Declarative(; memory = chunks)
    # initialize the ACTR object and add parameters
    actr = ACTR(; declarative = memory, parms..., δ)
    # initialize an array of data
    np = NamedTuple{(:position, :resp), Tuple{Int64, Int64}}
    data = Array{np, 1}(undef, n_items)
    #indices for each chunk. -100 is a retrieval failure
    ω = [1:n_items; -100]
    for i = 1:n_items
        # compute retrieval probabilities θ for retrieval request current position i 
        # and non-retrieved chunks
        θ, request = retrieval_probs(actr; position = i, retrieved = [false])
        # select a random chunk index weighted by retrieval probabilities
        resp = sample(ω, Weights(θ))
        # get the retrieved chunk
        chunk = request[resp]
        # set retrieved chunk to retrieved = true
        modify!(chunk.slots, retrieved = true)
        # add NamedTuple for current position and position of retrieved chunk
        data[i] = (position = i, resp = chunk.slots.position)
    end
    return data
end

function penalty(actr, chunk; criteria...)
    slots = chunk.slots
    p = 0.0
    δ = actr.parms.δ
    for (c, v) in criteria
        p += δ * penalize(slots[c], v)
    end
    return p
end

penalize(v1::Array{Bool, 1}, v2::Array{Bool, 1}) = sum(@. abs(v1 - v2) * 1.0)
penalize(v1, v2) = abs(v1 - v2)

function computeLL(parms, all_data, n_items; δ)
    act = zero(typeof(δ))
    chunks = populate_memory(n_items, act)
    # populate declarative memory with chunks
    memory = Declarative(; memory = chunks)
    # initialize the ACTR object and add parameters
    actr = ACTR(; declarative = memory, parms..., δ)
    # initialize log likelihood
    LL = 0.0
    # loop over each block of data
    for data in all_data
        # reset retrieval slot to false for all chunks
        reset_memory!(chunks)
        # compute log likelhood of block data
        LL += block_LL(actr, data)
    end
    return LL
end

function block_LL(actr, data)
    # initialize block log likelihood for block
    LL = 0.0
    # iterate over each block trial
    for k in data
        # get the retrieved trucked
        chunk = get_chunks(actr; position = k.resp, retrieved = [false])
        # compute the probability of the retrieved chunk
        θᵣ, _ = retrieval_prob(actr, chunk; position = k.position, retrieved = [false])
        # set retrieved = true for the retrieved chunk
        modify!(chunk[1].slots, retrieved = true)
        # increment the log likelihood LL
        LL += log(θᵣ)
    end
    return LL
end

function reset_memory!(chunks)
    for c in chunks
        modify!(c.slots, retrieved = false)
    end
    return nothing
end

function transpostions(parms, n_items; δ)
    data = simulate(parms, n_items; δ)
    positions = map(x -> x.resp, data)
    return @. abs([1:n_items;] - positions)
end

function serial_position(parms, n_items; δ)
    data = simulate(parms, n_items; δ)
    positions = map(x -> x.resp, data)
    return @. 1:n_items .== positions
end
