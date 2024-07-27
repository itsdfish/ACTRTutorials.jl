using Parameters, Distributions, StatsFuns
import Distributions: logpdf, rand, loglikelihood

mutable struct Grouped{T1, T2} <: ContinuousUnivariateDistribution
    δ::T1
    fixed_parms::T2
    n_groups::Int
    n_positions::Int
end

Grouped(; δ, fixed_parms, n_groups = 3, n_positions = 3) =
    Grouped(δ, fixed_parms, n_groups, n_positions)

loglikelihood(d::Grouped, Data::Vector{Vector{Int64}}) = logpdf(d, Data)

function logpdf(d::Grouped, Data::Vector{Vector{Int64}})
    LL = computeLL(Data, d.fixed_parms, d.n_groups, d.n_positions; δ = d.δ)
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
    # initialize data for all trials
    data = Int[]
    # inhabition of return: exclude retrieved chunks
    retrieved = [false]
    # loop over all groups in asecnding order
    for group = 1:n_groups
        # loop over all positions within a group in ascending order
        for position = 1:n_positions
            # retrieve chunk given group and position indices
            # exclude retrieved chunks
            probs, r_chunks = retrieval_probs(actr; group, position, retrieved)
            n_probs = length(probs)
            idx = sample(1:n_probs, Weights(probs))
            # ommit response on retrieval failure
            if idx < n_probs
                # set chunk to retrieved = true for inhabition of return
                chunk = r_chunks[idx]
                chunk.slots.retrieved[1] = true
                # record the number value of the retrieved chunk
                push!(data, chunk.slots.number)
            end
        end
    end
    return data
end

function populate_memory(act = 0.0, n_groups = 3, n_positions = 3)
    cnt = 0
    chunks = [
        Chunk(; act, group = i, number = cnt += 1, position = j, retrieved = [false])
        for i = 1:n_groups for j = 1:n_positions
    ]
    return chunks
end

function computeLL(Data, fixed_parms, n_groups = 3, n_positions = 3; δ)
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
        # marginalize over all possible orders of retrievals and retrieval failrues for a given block of trials
        LL += computeLLMultisets(actr, data, n_groups, n_positions; δ)
        # reset all chunks to retrieved = false
    end
    return LL
end

function computeLLMultisets(actr, data, n_groups = 3, n_positions = 3; δ)
    # initialize log likelihood
    LL = 0.0
    n_items = n_groups * n_positions
    n = length(data)
    # n retrievals coded as false plus (n_items - n) retrieval failures true
    indicators = [fill(false, n); fill(true, n_items - n)]
    # all possible orders of n retrievals and (n_items - n ) retrieval failures
    orders = multiset_permutations(indicators, length(indicators))
    # a vector of log likelihoods to be marginalized
    LLs = zeros(typeof(δ), length(orders))
    # compute the log likelihoods of the data across all possible orders of 
    # retrievals and retrieval failures
    for (i, order) in enumerate(orders)
        LLs[i] = computeLLBlock(actr, data, order, n_groups, n_positions)
        reset_memory!(actr)
    end
    # return the marginal log likelihood
    return logsumexp(LLs)
end

function computeLLBlock(actr, data, order, n_groups = 3, n_positions = 3)
    # inhabition of return: exclude retrieved chunks
    retrieved = [false]
    # index for retrieval attempts
    r_idx = 0
    # index for the obseved data
    data_idx = 0
    LL = 0.0
    for group = 1:n_groups
        for position = 1:n_positions
            r_idx += 1
            # log likelihood of retrieval failure given retrieval request for group and position indices
            p, r_chunks = retrieval_probs(actr; group, position, retrieved)
            # compute log likelihood of retrieval failure if order is true
            if order[r_idx] == true
                # log likelihood of retrieval failure
                LL += log(p[end])
            else
                # compute the log likelihood of retrieving chunk associated with next response
                # increment data index
                data_idx += 1
                # get the index associated with retrieved chunk
                chunk_idx = find_index(r_chunks; number = data[data_idx])
                # set the retrieved chunk to retrieved = true for inhabition of return 
                r_chunks[chunk_idx].slots.retrieved[1] = true
                # log likelihood of retrieving chunk associated with response 
                LL += log(p[chunk_idx])
            end
        end
    end
    return LL
end

function reset_memory!(actr)
    chunks = actr.declarative.memory
    map(x -> x.slots.retrieved[1] = false, chunks)
    return nothing
end

function multisets(r, f)
    x = [fill("r", r); fill("f", f)...]
    return multiset_permutations(x, length(x))
end
