using Parameters, StatsBase, Random, Distributed

function simulate(stimuli, fixed_parms; args...)
    # populate memory with addition facts
    chunks = populate_memory()
    # set blc parameters for each chunk
    set_baselevels!(chunks)
    # add parameters and chunks to declarative memory
    memory = Declarative(;memory=chunks)
    # add declarative memory to ACTR object
    actr = ACTR(;declarative=memory, fixed_parms..., args...)
    ter = get_parm(actr, :ter)
    N = length(stimuli)
    data = fill(0, N)
    cur_time = 120.0
    rt = 0.0
    for (i,s) in enumerate(stimuli)
        # retrieval probabilities θs given request s
        chunk = retrieve(actr, cur_time; s...)
        if isempty(chunk)
            data[i] = -100
            rt = compute_RT(actr, chunk) 
        else 
            _chunk = chunk[1]
            rt = compute_RT(actr, _chunk)
            data[i] = _chunk.slots.sum
            update_chunk!(_chunk, cur_time + ter + rt)
        end
        cur_time += max(10.0, rt + ter + 1) 
    end
    return data
end

function sim_fun(actr, chunk; request...)
    slots = chunk.slots
    p = 0.0; δ = actr.parms.δ
    for (c,v) in request
        p += .1 * δ * abs(slots[c] - v)
    end
    return p
end

function populate_memory(act=0.0, N=25)
    chunks = [Chunk(;num1=num1,num2=num2,
        sum=num1 + num2,act=act, N=N, recent=[100.0]) for num1 in 0:5
        for num2 in 0:5]
    pop!(chunks)
    return chunks
end

function set_baselevels!(chunks)
    for chunk in chunks
        if chunk.slots.sum < 5
            chunk.bl = .65
        end
    end
    return nothing
end

function loglike(δ, stimuli, fixed_parms, data; n_sim=1000)
    sim_data = pmap(_->simulate(stimuli, fixed_parms; δ), 1:n_sim)
    counts = mapreduce(x-> x .=== data, +, sim_data)
    p = counts/n_sim
    @. p = max(p, 1e-10) 
    return sum(log.(p))
end

function response_histogram(df, stimuli; kwargs...)
    vals = NamedTuple[]
    for (num1,num2) in stimuli
        idx = @. ((df[:,:num1] == num1 ) & (df[:,:num2] == num2)) | ((df[:,:num1] == num2) & (df[:,:num2] == num1))
        subdf = df[idx,:]
        str = string(num1, "+", num2)
        v = filter(x -> x != -100, subdf[:,:resp])
        push!(vals, (title = str, data = v))
    end
    p = histogram(layout=(2,3), leg=false, xlims=(0,10), xlabel="Response",ylabel="Proportion",
        size=(600,600), xaxis=font(12), yaxis=font(12), titlefont=font(12), grid=false; kwargs...)
    for (i,v) in enumerate(vals)
        histogram!(p, v.data, subplot=i, title=v.title, bar_width=1, color=:grey, grid=false,
        normalize=:probability, ylims=(0,1))
    end
    return p
end