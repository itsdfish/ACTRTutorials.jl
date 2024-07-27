function simulate(d, blc, delay, N; fixed_parms...)
    cur_time = N * 2.0
    # create a chunk
    chunk = Chunk(; N, recent = [cur_time])
    # add the chunk to declarative memory
    memory = Declarative(; memory = [chunk])
    # create ACTR object and pass parameters
    actr = ACTR(; declarative = memory, d, blc, fixed_parms...)
    # random wait time
    cur_time += delay
    # time to encode stimulus and select retrieval production
    cur_time += 0.05 + 0.085 + 0.05
    # retrieval probability p
    p, _ = retrieval_prob(actr, chunk, cur_time)
    return rand(Bernoulli(p))
end

function loglike(d, blc, delay, N, retrieved; fixed_parms...)
    # each practice is 2 seconds
    cur_time = N * 2.0
    # create a chunk
    chunk = Chunk(; N, recent = [cur_time])
    # add the chunk to declarative memory
    memory = Declarative(; memory = [chunk])
    # create ACTR object and pass parameters
    actr = ACTR(; declarative = memory, d, blc, fixed_parms...)
    # random wait time
    cur_time += delay
    # time to encode stimulus and select retrieval production
    cur_time += 0.05 + 0.085 + 0.05
    # compute retrieval probability r and retrieval failure probability f
    r, f = retrieval_prob(actr, chunk, cur_time)
    p = retrieved ? r : f
    # avoid -Inf values
    p = max(p, eps())
    p = min(1 - eps(), p)
    return log(p)
end

function all_LL(choices, d, blc, delay, N; fixed_parms...)
    LL = 0.0
    for choice in choices
        LL += loglike(d, blc, delay, N, choice; fixed_parms...)
    end
    return LL
end

function activation_dynamics(delays, N; blc, d, fixed_parms...)
    # each practice is 2 seconds
    cur_time = N * 2.0
    # create a chunk
    chunk = Chunk(; N, recent = [cur_time])
    # add the chunk to declarative memory
    memory = Declarative(; memory = [chunk])
    # create ACTR object and pass parameters
    actr = ACTR(; declarative = memory, d, blc, fixed_parms...)
    activations = fill(0.0, length(delays))
    for (s, delay) in enumerate(delays)
        compute_activation!(actr, cur_time + delay)
        activations[s] = chunk.act
    end
    return activations
end
