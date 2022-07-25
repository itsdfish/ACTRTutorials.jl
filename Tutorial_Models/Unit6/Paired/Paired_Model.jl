using Parameters, Random
import Distributions: logpdf, loglikelihood

struct Paired{T1,T2} <: ContinuousUnivariateDistribution
    d::T1
    parms::T2
end

Paired(;d, parms) = Paired(d, parms)

loglikelihood(d::Paired, data::Array{<:NamedTuple,1}) = logpdf(d, data)

function logpdf(d::Paired, data::Array{<:NamedTuple,1})
    LL = computeLL(data, d.parms; d=d.d)
    return LL
end

function sample_stimuli(stimuli, N)
    return stimuli[1:N]
end

function simulate(all_stimuli, fixed_parms, n_blocks, n_trials, deadline=5.0, isi=5.0; d)
    # sample unique stimului for the simulation
    stimuli = sample_stimuli(all_stimuli, n_trials)
    # initialize declarative memory
    chunks = [Chunk(word=:_,number=1)] |> empty!
    memory = Declarative(;memory=chunks)
    # create act-r object
    actr = ACTR(;declarative=memory, fixed_parms..., d)
    # preallocate data
    data = Array{NamedTuple,1}()
    # initialize time
    cur_time = 0.0
    # iterate through all blocks
    for block in 1:n_blocks
        # randomize stimulus order
        shuffle!(stimuli)
        temp,cur_time = simulate_block(actr, stimuli, cur_time, block, deadline, isi)
        push!(data, temp...)
    end
    return vcat(data...)
end

function simulate_block(actr, stimuli, cur_time, block, deadline=5.0, isi=5.0)
    # initialize data for the current block
    data = Array{NamedTuple,1}()
    for stimulus in stimuli
        temp = simulate_trial(actr, stimulus, cur_time, block, deadline)
        push!(data, temp)
        # 5 second response deadline + additional 5 seconds
        cur_time += (deadline + isi) 
    end
    return data,cur_time
end

function simulate_trial(actr, stimulus, cur_time, block, deadline=5.0)
    N=0;L=0.0;recent=Float64[];retrieved=:failed;time_created=0.0
    trial_start = cur_time
    # encode and conflict resolution time
    e_time = 0.085 + 0.05 + 0.05
    cur_time += e_time
    # retrieve paired associate based on word
    chunk = retrieve(actr, cur_time; word=stimulus.word)
    # retrieval time 
    r_time = compute_RT(actr, chunk)
    # add encoding time and retrieval time to reaction time
    rt = e_time + r_time + .05 + 0.300
    # interrupt retrieval if exceeds deadline
    if rt > deadline
        # set to truncated value
        rt = deadline
        # advance current time to deadline
        cur_time  = trial_start + deadline
        retrieved = :truncated
        # retrieval info if no chunk is retrieved
        chunk = get_chunks(actr; stimulus...)
        if !isempty(chunk)
            # get chunk state at time of retrieval
            @unpack N,L,recent,time_created = chunk[1]
            recent = copy(recent)
        end
    elseif !isempty(chunk)
        # add retreival time to current time
        cur_time += r_time 
        # next conflict resolution
        cur_time += .05
        retrieved = :retrieved
        # get chunk state at time of retrieval
        @unpack N,L,recent,time_created = chunk[1]
        recent = copy(recent)
        # add chunk if new, or update N and time stamps
        add_chunk!(actr, cur_time; stimulus...)
    else
        # add retreival failure time to current time
        cur_time += r_time 
        # next conflict resolution
        cur_time += .05
        # retrieval info if no chunk is retrieved
        chunk = get_chunks(actr; stimulus...)
        if !isempty(chunk)
            # get chunk state at time of retrieval
            @unpack N,L,recent,time_created = chunk[1]
            recent = copy(recent)
        end
    end
    # feedback is always presented after the deadline
    cur_time = trial_start + deadline
    #encode feedback
    cur_time += (0.05 + 0.05 + 0.085)
    # add chunk: create new chunk if does not exist, otherwise increment N and add time stamp
    add_chunk!(actr, cur_time; stimulus...)
    data = (stimulus=stimulus,N=N,L=L,block=block,time_created=time_created,
        recent=recent,rt=rt,retrieved=retrieved)
    return data
end

function computeLL(data, fixed_parms; d)
    chunk = Chunk(act=zero(d))
    # initialize declarative memory
    memory = Declarative(;memory=[chunk])
    # create ACTR object with declarative memory and parameters
    actr = ACTR(;declarative=memory, fixed_parms..., d)
    # extract parameters
    @unpack s,lf,τ,ter=actr.parms
    # do not add noise activation values
    actr.parms.noise = false
    σ = s * pi / sqrt(3)
    # initialize log likelihood
    LL = 0.0
    for k in data
        cur_time = k.L + k.time_created
        if k.retrieved == :retrieved
            # reproduce the memory state right before retrieval
            modify!(chunk; N=k.N, recent=k.recent, time_created=k.time_created)
            # compute activation
            compute_activation!(actr, chunk, cur_time)
            # adjust activation values for latency factor
            v = [chunk.act,τ] .- log(lf)
            # log normal race distribution object
            dist = LNR(-v, σ, ter)
            # compute log likelihood of retrieving chunk at time k.rt
            LL += logpdf(dist, 1, k.rt)
        elseif k.retrieved == :truncated
            # special case in which chunk has not been created yet
            if k.N == 0
                v = τ .- log(lf)
                # compute log likelihood of failing to respond within deadline
                LL += logccdf(LogNormal(-v, σ), 5)
            else
                # reproduce the memory state right before retrieval
                modify!(chunk; N=k.N, recent=k.recent, time_created=k.time_created)
                # compute activation
                compute_activation!(actr, chunk, cur_time)
                # adjust activation values for latency factor
                v = [chunk.act,τ] .- log(lf)
                # compute log likelihood of failing to respond within deadline
                LL += logccdf(LogNormal(-v[1], σ), 5) + logccdf(LogNormal(-v[2], σ), 5)
            end
        else
            # retrieval failure for novel pair
            if k.N == 0
                dist = LNR(-[τ-log(lf)], σ, ter)
                LL += logpdf(dist, 1, k.rt)
            else
                # reproduce the memory state immediately prior to retrieval
                modify!(chunk; N=k.N, recent=k.recent, time_created=k.time_created)
                # compute activation 
                compute_activation!(actr, chunk, cur_time)
                # mean activation values
                v = [chunk.act,τ] .- log(lf)
                # log normal race distribution object
                dist = LNR(-v, σ, ter)
                # compute log likelihood of failing to retrieve chunk at time k.rt
                LL += logpdf(dist, 2, k.rt)
            end
        end
    end
    return LL
end

function show_learning(all_stimuli, parms, n_blocks=8, n_trials=20; d)
    data = simulate(all_stimuli, parms, n_blocks, n_trials; d)
    df = DataFrame(data)
    groups = groupby(df, [:block,:retrieved])
    return combine(groups, :rt=>mean=>:rt)
end
