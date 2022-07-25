using Parameters, Distributions, StatsFuns
import Distributions: logpdf, rand, loglikelihood
struct Retrieval{T1,T2} <: ContinuousUnivariateDistribution
    d::T1
    parms::T2
end

Retrieval(;d, parms) = Retrieval(d, parms)

loglikelihood(d::Retrieval, data::Array{<:NamedTuple,1}) = logpdf(d, data)

function logpdf(d::Retrieval, data::Array{<:NamedTuple,1})
    return computeLL(d.parms, data; d=d.d)
end

function simulate(parms, n_trials; d)
    # create a chunk
    chunk = Chunk()
    # add the chunk to declarative memory
    memory = Declarative(;memory=[chunk])
    # create ACTR object and pass parameters
    actr = ACTR(;declarative=memory, parms..., d=d)
    # initialize data array
    data = Array{NamedTuple,1}(undef, n_trials)
    # initialize response variable and initialize current time at 0.0
    cur_time = 0.0
    for i in 1:n_trials
        # random wait time
        cur_time += rand(Uniform(5, 15))
        # time to encode stimulus and select retrieval production
        cur_time += 0.05 + 0.085 + 0.05
        # retrieve a chunk
        requested = retrieve(actr, cur_time)
        # compute retrieval time
        rt = compute_RT(actr, requested)
        # incorrect if empty, correct otherwise
        resp = isempty(requested) ? false : true
        # extract information to reconstruct activation
        @unpack N,L,lags,recent = chunk
        # add information to data for trial i
        data[i] = (recent = copy(recent),lags = lags,L = L,N = N,resp = resp)
        # add retrieval time 
        cur_time += rt
        # respond 
        cur_time += 0.05 + 0.06
        # update chunk with the time
        # time to encode feedback and update
        cur_time += 0.05 + 0.085 + 0.05 
        # reinforce the memory
        update_chunk!(chunk, cur_time)
    end
    return data
end

function computeLL(parms, data; d)
    act = zero(typeof(d))
    # create chunk
    chunk = Chunk(;act=act)
    # add chunk to declarative memory
    memory = Declarative(;memory=[chunk])
    # create ACTR object and pass parameters
    actr = ACTR(;declarative=memory, parms..., d=d)
    LL = act
    for k in data
        # add time stamps and lags to chunk
        modify!(chunk; N=k.N, lags=k.lags, recent=k.recent)
        # compute retrieval probability r and retrieval failure probability f
        r,f = retrieval_prob(actr, chunk, k.L)
        # compute log likelihood for each response (resp = true -> correct, resp = false->incorrect)
        if k.resp
            LL += log(r)
        else
            LL += log(f)
        end
    end
    return LL
end

function learning_block(data, bsize=10)
    resp = map(x -> x.resp, data)
    N = Int(length(data) / bsize)
    return [mean(resp[((i - 1) * bsize + 1):(i * bsize)]) for i in 1:N]
end

function activation_dynamics(time_steps, retrieval_times, n_retrievals=10, duration=60, d=.5)
    sort!(retrieval_times)
    chunk = Chunk()
    declarative = Declarative(memory=[chunk])
    actr = ACTR(declarative=declarative, bll=true, d=d)
    time_stamp = retrieval_times[1]
    activations = fill(0.0, length(time_steps))
    cnt = 1
    for (s,t) in enumerate(time_steps)
        compute_activation!(actr, t)
        activations[s] = chunk.act
        if (time_stamp < t) && (cnt <= n_retrievals)
            update_chunk!(chunk, time_stamp)
            time_stamp = retrieval_times[cnt]
            cnt += 1
        end
    end
    return activations
end