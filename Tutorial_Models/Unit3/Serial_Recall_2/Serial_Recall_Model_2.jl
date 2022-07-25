using Parameters, Distributions, StatsFuns, StatsBase
import Distributions: logpdf, rand, loglikelihood

struct SerialRecall{T1,T2,T3,T4} <: ContinuousUnivariateDistribution
    δ::T1
    τ::T2
    d::T3
    parms::T4
    n_trials::Int
end

Broadcast.broadcastable(x::SerialRecall) = Ref(x)

SerialRecall(;δ, τ, d, parms, n_trials) = SerialRecall(δ, τ, d, parms)

loglikelihood(d::SerialRecall, data::Array{<:Array{<:NamedTuple,1},1}) = logpdf(d, data)

function logpdf(d::SerialRecall ,data::Array{<:Array{<:NamedTuple,1},1})
    return computeLL(d.parms, data, d.n_trials; δ=d.δ, τ=d.τ, d=d.d)
end

function initialize_memory(act=0.0)
    chunks = [Chunk(;act=act, position=1, retrieved=[false])]
    return typeof(chunks)()
end

function populate_memory(n_items, act=0.0)
    chunks = [Chunk(;act=act, position=i, retrieved=[false]) for i in 1:n_items]
    return chunks
end

function simulate(parms, n_study, n_items; δ, τ, d)
    chunks = initialize_memory()
    # populate declarative memory with chunks
    memory = Declarative(;memory=chunks)
    # initialize the ACTR object
    actr = ACTR(;declarative=memory, parms..., δ, τ, d)
    # initialize time
    cur_time = 0.0
    # study list items
    cur_time = simulate_study!(actr, n_study, n_items, cur_time)
    # add 3 second delay between study and test phases
    cur_time += 1.0
    return simulate_test!(actr, n_items, cur_time)
end

function simulate_study!(actr, n_study, n_items, cur_time)
    # stimulus presentation rate
    p_rate = 2.0
    encoding_time = 0.5
    rehersal_time = 0.5
    chunks = actr.declarative.memory
    # repeat study list
    for rep ∈ 1:n_study
        # loop through each chunk
        for m ∈ 1:n_items
            # increment time
            cur_time += encoding_time
            add_chunk!(actr, cur_time; position=m, retrieved=[false])
            # rehersal index
            r = 1
            # reherse from r ∈ 1:m while time remaining
            while mod(cur_time, p_rate) != 0 
                # increment time
                cur_time += rehersal_time
                # println(cur_time, " m ", m, " r ", r)
                # update chunk
                update_chunk!(chunks[r], cur_time)
                #reset reloop through if necessary
                r  = r == m ? 1 : r += 1
            end
        end
    end
    return cur_time
end

function simulate_test!(actr, n_items, cur_time)
    parms = actr.parms
    data = Array{NamedTuple,1}(undef,n_items)
    chunks = actr.declarative.memory
    chunk = deepcopy(chunks[1])
    rt = 0.0
    for i ∈ 1:n_items
        cur_time += 0.05
        # compute retrieval probabilities θ for retrieval request current position i and non-retrieved chunks
        θ,request = retrieval_probs(actr, cur_time; position=i, retrieved=[false])
        n_choices = length(θ)
        # select a random chunk index weighted by retrieval probabilities
        chunk_id = sample(1:n_choices, Weights(θ))
        compute_activation!(actr, cur_time; position=i, retrieved=[false])
        if chunk_id == n_choices
            # guessing process
            chunk = rand(request)
            rt = compute_RT(actr, Chunk[])
        else
            # retrieval process
            chunk = request[chunk_id] 
            rt = compute_RT(actr, chunk)
        end
        # add state, position, respose and current time to data
        data[i] = (state = deepcopy(chunks), position=i, resp=chunk.slots.position,
             cur_time=cur_time)
        cur_time += rt
        # set retrieved chunk to retrieved = true
        modify!(chunk.slots, retrieved=true)
        # time to respond
        cur_time += (0.05 + 0.5)
    end
    return data
end

function penalty(actr, chunk; criteria...)
    slots = chunk.slots
    p = 0.0; δ = actr.parms.δ
    for (c,v) in criteria
        p += δ * penalize(slots[c], v)
    end
    return p
end

penalize(v1::Array{Bool,1}, v2::Array{Bool,1}) = sum(@. abs(v1-v2)*1.0)
penalize(v1, v2) = abs(v1 - v2)

function computeLL(parms, all_data, n_items; δ, τ, d)
    act = zero(typeof(δ))
    chunks = populate_memory(n_items, act)
    # populate declarative memory with chunks
    memory = Declarative(;memory=chunks)
    # initialize the ACTR object
    actr = ACTR(;declarative=memory, parms..., δ, τ, d)
    # initialize time
    cur_time = 0.0
    # initialize log likelihood
    LL = 0.0
    # loop over each block of data
    for data in all_data
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
        set_state!(actr, k.state)
        request = retrieval_request(actr; position=k.position, retrieved=[false])
        # get the retrieved trucked
        chunk = get_chunks(actr; position=k.resp, retrieved=[false])
        # compute the probability of the retrieved chunk and retrieval failure
        θᵣ,rf = retrieval_prob(actr, chunk, k.cur_time; position=k.position, retrieved=[false])
        θ = θᵣ + rf * (1 / length(request))
        # increment the log likelihood LL
        LL += log(θ)
    end
    return LL
end

function set_state!(actr, data_chunks)
    chunks = actr.declarative.memory
    for (c,v) in zip(chunks,data_chunks)
        modify!(c; N=v.N, lags=v.lags, recent=v.recent)
        modify!(c.slots, retrieved=v.slots.retrieved[1])
    end
    return nothing
end

function transpostions(parms, n_study, n_items; δ, τ, d)
    data = simulate(parms, n_study, n_items; δ, τ, d)
    positions = map(x->x.resp, data)
    return @. abs([1:n_items;] - positions)
end

function serial_position(parms, n_study, n_items; δ, τ, d)
    data = simulate(parms, n_study, n_items; δ, τ, d)
    positions = map(x->x.resp, data)
    return @. 1:n_items .== positions
end
