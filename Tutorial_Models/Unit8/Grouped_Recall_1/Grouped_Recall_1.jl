using Parameters, Distributions, StatsFuns
import Distributions: logpdf, rand, loglikelihood

mutable struct Grouped{T1, T2} <: ContinuousUnivariateDistribution
    δ::T1
    fixed_parms::T2
end

Grouped(; δ, fixed_parms) = Grouped(δ, fixed_parms)

loglikelihood(d::Grouped, Data::Array{<:T, 1}) where {T <: Array{<:NamedTuple, 1}} =
    logpdf(d, Data)

function logpdf(d::Grouped, Data::Array{<:T, 1}) where {T <: Array{<:NamedTuple, 1}}
    LL = computeLL(Data, d.fixed_parms; δ = d.δ)
    return LL
end

function sim_fun(actr, chunk; criteria...)
    slots = chunk.slots
    p = 0.0
    δ = actr.parms.δ
    for (c, v) in criteria
        if (c == :retrieved) || (c == :isa)
            continue
        else
            p += δ * rel_diff(slots[c], v)
        end
    end
    return p
end

function rel_diff(v1, v2)
    return abs(v1 - v2) / max(v1, v2)
end

function simulate(n_groups = 3, n_positions = 3; δ, fixed_parms...)
    # initialize memory with chunks representing all stimuli
    chunks = populate_memory()
    # add chunks to declarative memory
    memory = Declarative(; memory = chunks)
    # create ACT-R model object
    actr = ACTR(; declarative = memory, δ, fixed_parms...)
    N = n_groups * n_positions
    # initialize data for all trials
    data = Array{NamedTuple, 1}(undef, N)
    cnt = 0
    # inhabition of return: exclude retrieved chunks
    retrieved = [false]
    # loop over all groups in asecnding order
    for group = 1:n_groups
        # loop over all positions within a group in ascending order
        for position = 1:n_positions
            cnt += 1
            # retrieve chunk given group and position indices
            # exclude retrieved chunks
            probs, r_chunks = retrieval_probs(actr; group, position, retrieved)
            n_probs = length(probs)
            idx = sample(1:n_probs, Weights(probs))
            if idx == n_probs
                # code retrieval failure as -100
                data[cnt] = (group = group, position = position, resp = -100)
            else
                # set chunk to retrieved = true for inhabition of return
                chunk = r_chunks[idx]
                chunk.slots.retrieved[1] = true
                # record the number value of the retrieved chunk
                data[cnt] = (group = group, position = position, resp = chunk.slots.number)
            end
        end
    end
    return vcat(data...)
end

function populate_memory(act = 0.0, n_groups = 3, n_positions = 3)
    cnt = 0
    chunks = [
        Chunk(; act, group = i, number = cnt += 1, position = j, retrieved = [false])
        for i = 1:n_groups for j = 1:n_positions
    ]
    return chunks
end

function computeLL(Data, fixed_parms; δ)
    T = typeof(δ)
    act = zero(T)
    # initialize chunks
    chunks = populate_memory(act)
    # add chunks to declarative memory
    memory = Declarative(; memory = chunks)
    # create ACT-R object with declarative memory and parameters
    actr = ACTR(; declarative = memory, fixed_parms..., δ)
    LL::T = 0.0
    # loop over each block of trials
    for data in Data
        # log likelihood of block of trials
        LL += computeLLBlock(data, actr; δ)
        # reset all chunks to retrieved = false
        reset_memory!(actr)
    end
    return LL
end

function computeLLBlock(data, actr; δ)
    # initialize log likelihood
    LL = 0.0
    # inhabition of return: exclude retrieved chunks
    retrieved = [false]
    for k in data
        if k.resp == -100
            # log likelihood of retrieval failure given retrieval request for group and position indices
            p, _ = retrieval_probs(actr; group = k.group, position = k.position, retrieved)
            LL += log(p[end])
        else
            # get retrieved chunk
            chunk = get_chunks(actr; number = k.resp)
            # log likelihood of retrieving chunk given retrieval request for group and position indices
            p, _ = retrieval_prob(
                actr,
                chunk;
                group = k.group,
                position = k.position,
                retrieved
            )
            # set the retrieved chunk to retrieved = true for inhabition of return 
            chunk[1].slots.retrieved[1] = true
            LL += log(p)
        end
    end
    return LL
end

function reset_memory!(actr)
    chunks = actr.declarative.memory
    map(x -> x.slots.retrieved[1] = false, chunks)
    return nothing
end

function displacement(data)
    responses = map(x -> x.resp, data)
    answers = 1:length(responses)
    map!(i -> responses[i] == -100 ? i : responses[i], responses, 1:length(responses))
    return @. abs(responses - answers)
end
