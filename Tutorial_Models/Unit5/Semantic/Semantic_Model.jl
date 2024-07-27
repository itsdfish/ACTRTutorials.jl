using Parameters, StatsBase, NamedTupleTools

import Distributions: logpdf, loglikelihood

struct Semantic{T1, T2} <: ContinuousUnivariateDistribution
    blc::T1
    parms::T2
end

loglikelihood(d::Semantic, data::Array{<:NamedTuple, 1}) = logpdf(d, data)

Semantic(; blc, parms) = Semantic(blc, parms)

function populate_memory(act = 0.0)
    chunks = [
        Chunk(object = :shark, attribute = :dangerous, value = :True, act = act),
        Chunk(object = :shark, attribute = :locomotion, value = :swimming, act = act),
        Chunk(object = :shark, attribute = :category, value = :fish, act = act),
        Chunk(object = :salmon, attribute = :edible, value = :True, act = act),
        Chunk(object = :salmon, attribute = :locomotion, value = :swimming, act = act),
        Chunk(object = :salmon, attribute = :category, value = :fish, act = act),
        Chunk(object = :fish, attribute = :breath, value = :gills, act = act),
        Chunk(object = :fish, attribute = :locomotion, value = :swimming, act = act),
        Chunk(object = :fish, attribute = :category, value = :animal, act = act),
        Chunk(object = :animal, attribute = :moves, value = :True, act = act),
        Chunk(object = :animal, attribute = :skin, value = :True, act = act),
        Chunk(object = :canary, attribute = :color, value = :yellow, act = act),
        Chunk(object = :canary, attribute = :sings, value = :True, act = act),
        Chunk(object = :canary, attribute = :category, value = :bird, act = act),
        Chunk(object = :ostritch, attribute = :flies, value = :False, act = act),
        Chunk(object = :ostritch, attribute = :height, value = :tall, act = act),
        Chunk(object = :ostritch, attribute = :category, value = :bird, act = act),
        Chunk(object = :bird, attribute = :wings, value = :True, act = act),
        Chunk(object = :bird, attribute = :locomotion, value = :flying, act = act),
        Chunk(object = :bird, attribute = :category, value = :animal, act = act)
    ]
    return chunks
end

function get_stimuli()
    stim = NamedTuple[]
    push!(stim, (object = :canary, category = :bird, ans = :yes))
    push!(stim, (object = :canary, category = :animal, ans = :yes))
    return vcat(stim...)
end

function logpdf(d::Semantic, data::Array{<:NamedTuple, 1})
    LL = computeLL(d.parms, data; blc = d.blc)
    return LL
end

function simulate(fixed_parms, stimulus, n_reps; blc)
    # populate declarative memory
    chunks = populate_memory()
    # generate declarative memory object
    memory = Declarative(; memory = chunks)
    # generate ACTR object
    actr = ACTR(; declarative = memory, fixed_parms..., blc)
    # the number of correct responses
    k = 0
    # count the number of correct answers, k
    for rep = 1:n_reps
        k += simulate_trial(actr, stimulus)
    end
    # return data that constains stimulus information, number of trials, 
    # and correct answers
    return (stimulus..., N = n_reps, k = k)
end

function simulate_trial(actr, stimulus)
    retrieving = true
    # create memory probe or retrieval request
    probe = stimulus
    chunks = actr.declarative.memory
    # k = 1 if answer is "yes", 0 otherwise
    k = 0
    while retrieving
        # generate retrieval probabilities
        p, _ = retrieval_probs(actr; object = probe.object, attribute = :category)
        # sample a chunk index proportional to retrieval probabilities
        idx = sample(1:length(p), weights(p))
        # Last element corresponds to a retrieval failure
        # stop retrieval processes
        if idx == length(p)
            retrieving = false
            # retrieved chunk matches retrieval request, stop retrieving
            # and set k = 1 for "yes" response
        elseif direct_verify(chunks[idx], probe)
            retrieving = false
            k += 1
            # perform another retrieval with category chaining
            # modify the retrieval request based on the retrieved chunk
        elseif chain_category(chunks[idx], probe)
            probe = delete(probe, :object)
            probe = (object = chunks[idx].slots.value, probe...)
            # no chunks match, stop retrieving and respond "no" with k = 0
        else
            retrieving = false
        end
    end
    return k
end

function simulate_exact(fixed_parms, stimulus, n_reps; blc)
    chunks = populate_memory()
    memory = Declarative(; memory = chunks)
    actr = ACTR(; declarative = memory, fixed_parms..., blc)
    k = 0
    for rep = 1:n_reps
        retrieving = true
        probe = stimulus
        while retrieving
            chunk = retrieve(actr; object = probe.object, attribute = :category)
            if isempty(chunk)
                retrieving = false
            elseif direct_verify(chunk[1], probe)
                retrieving = false
                k += 1
            elseif chain_category(chunk[1], probe)
                probe = delete(probe, :object)
                probe = (object = chunk[1].slots.value, probe...)
            else
                retrieving = false
            end
        end
    end
    return (stimulus..., N = n_reps, k = k)
end

"""
Answer yes via direct verification if retrieved chunk matches
probe on the object slot, the attribute slot equals category and the 
value slot matches the value of the probe's category slot
"""
function direct_verify(chunk, probe)
    return match(chunk, object = probe.object,
        value = probe.category, attribute = :category)
end

"""
Chain category if retrieved chunk matches
probe on the object slot, the attribute slot equals category and the 
value slot does not match the value of the probe's category slot
"""
function chain_category(chunk, probe)
    return match(chunk, ==, !=, ==, object = probe.object,
        value = probe.category, attribute = :category)
end

function initial_state(blc)
    s0 = zeros(typeof(blc), 5)
    s0[1] = 1
    return s0
end

function computeLL(fixed_parms, data; blc)
    act = zero(typeof(blc))
    # populate declarative memory
    chunks = populate_memory(act)
    # create declarative memory object
    memory = Declarative(; memory = chunks)
    # create act-r object
    actr = ACTR(; declarative = memory, fixed_parms..., blc)
    # create initial state vector
    s0 = initial_state(blc)
    LL = 0.0
    for d in data
        # create transition matrix
        tmat = transition_matrix(actr, d, blc)
        # compute probability of "yes"
        LL += probability_yes(tmat, s0, d)
    end
    return LL
end

function probability_yes(tmat, s0, d)
    z = s0' * tmat^3
    θ = z[4]
    # sometimes θ is nan because of exponentiation of activation
    return isnan(θ) ? (return -Inf) : logpdf(Binomial(d.N, θ), d.k)
end

"""
populatates transition matrix consisting of 5 states:
* `s1`: initial retrieval
* `s2 `: chain category 1
* `s3`: chain category 2
* `s4`: respond yes
* `s5`: respond no
"""
function transition_matrix(actr, stim, blc)
    chunks = actr.declarative.memory
    Nc = length(chunks) + 1
    probe::typeof(stim) = stim
    probe = stim
    N = 5
    # populate transition matrix
    tmat = zeros(typeof(blc), N, N)
    # compute retrieval probabilities, p
    p, _ = retrieval_probs(actr; object = get_object(probe), attribute = :category)
    # find indices of chunks associated with direct verification, category chaining and mismatching conditions
    direct_indices =
        find_indices(actr, object = get_object(probe), value = get_category(probe))
    chain_indices = find_indices(
        actr,
        ==,
        !=,
        ==,
        object = get_object(probe),
        value = get_category(probe),
        attribute = :category
    )
    mismatch_indices = setdiff(1:Nc, direct_indices, chain_indices)
    # use indices to compute probability of category chain, direct verification (yes), and mismatch (no)
    tmat[1, 2] = sum(p[chain_indices])
    tmat[1, 4] = sum(p[direct_indices])
    tmat[1, 5] = sum(p[mismatch_indices])
    # attempt to extract chunk associated with category chaining
    chain_chunk = get_chunks(actr, ==, !=, ==, object = get_object(probe),
        value = get_category(probe), attribute = :category)
    cnt = 1
    # continue the process above as long as category chaining can be performed.
    while !isempty(chain_chunk)
        cnt += 1
        probe = (object = get_chunk_value(chain_chunk[1]), delete(probe, :object)...)
        p, _ = retrieval_probs(actr; object = get_object(probe), attribute = :category)
        direct_indices =
            find_indices(actr, object = get_object(probe), value = get_category(probe))
        chain_indices = find_indices(
            actr,
            ==,
            !=,
            ==,
            object = get_object(probe),
            value = get_category(probe),
            attribute = :category
        )
        mismatch_indices = setdiff(1:Nc, direct_indices, chain_indices)
        tmat[cnt, 2] = sum(p[chain_indices])
        tmat[cnt, 4] = sum(p[direct_indices])
        tmat[cnt, 5] = sum(p[mismatch_indices])
        chain_chunk = get_chunks(actr, ==, !=, ==, object = get_object(probe),
            value = get_category(probe), attribute = :category)
    end
    # set self-transitions to 1 if row i sums to 0.0
    map(i -> sum(tmat[i, :]) == 0.0 ? (tmat[i, i] = 1.0) : nothing, 1:size(tmat, 2))
    return tmat
end

"""
Type-stable accessors
"""
get_object(x) = x.object
get_category(x) = x.category
get_chunk_value(x) = x.slots.value

function hit_rate(parms, stimulus, n_reps; blc)
    data = simulate(parms, stimulus, n_reps; blc = blc)
    return data.k / data.N
end

function parse_lisp_data(data)
    new_data = NamedTuple[]
    for row in eachrow(data)
        temp = (object = Symbol(row[1]), category = Symbol(row[2]),
            ans = Symbol(row[3]), N = row[4], k = row[5])
        push!(new_data, temp)
    end
    return vcat(new_data)
end

# https://discourse.julialang.org/t/eigenvalue-calculation-differs-between-julia-and-matlab/31185/17
# https://medium.com/@andrew.chamberlain/using-eigenvectors-to-find-steady-state-population-flows-cd938f124764
# https://www.mathworks.com/matlabcentral/answers/391167-steady-state-and-transition-probablities-from-markov-chain
# M = [.1 .2 0 .7 0;
#     .3 .4 .3 0 0;
#     0 0 .5 0 .5;
#     .2 .2 .2 .2 .2;
#     0 .1 .2 .3 .4]
#
# v = eigvecs(M')
# p = v[:,end]
# p = p/sum(p)
