using Parameters, StatsBase, Random, Distributed

function simulate(stimuli, fixed_parms; args...)
    N = length(stimuli)
    # initialize data
    data = Array{NamedTuple,1}(undef, N)
    # initialize ACT-R model 
    actr = initialize_model(fixed_parms; args...)
    # simulate each trial with stimulus s
    for (i,s) in enumerate(stimuli)
        response = simulate_trail(actr, s)
        data[i] = (s..., resp=response)
    end
    return data
end

function initialize_model(fixed_parms; args...)
   # populate memory with addition facts
   chunks = populate_memory()
   # set blc parameters for each chunk
   set_baselevels!(chunks)
   # add parameters and chunks to declarative memory
   memory = Declarative(;memory=chunks)
   # add declarative memory to ACTR object
   return actr = ACTR(;declarative=memory, fixed_parms..., args...)
end

function simulate_trail(actr, stimulus)
    # retrieve chunk using stimulus as retrieval request
    chunk = retrieve(actr; stimulus...)
    # return sum slot if retrieved, -100 for retrieval failure
    return isempty(chunk) ? -100 : chunk[1].slots.sum
end

function sim_fun(actr, chunk; request...)
    slots = chunk.slots
    p = 0.0; δ = actr.parms.δ
    for (c,v) in request
        p += .1 * δ * abs(slots[c] - v)
    end
    return p
end

function populate_memory(act=0.0)
    chunks = [Chunk(;num1=num1,num2=num2,
        sum=num1 + num2, act=act) for num1 in 0:5
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

function loglike(data, stimuli, fixed_parms, δ; n_sim=1000)
    # initialize the model with fixed_parms and δ
    actr = initialize_model(fixed_parms; δ)
    # temporary function for loglike_trial with two arguments
    f(s, x) = loglike_trial(actr, s, x, n_sim)
    # map the corresponding elements of stimuli and data to each processor
    LLs = pmap(f, stimuli, data)
    # sum the log likelihood. (Annotate return type because pmap is not type-stable)
    return sum(LLs)::Float64
end

function loglike_trial(actr, stimulus, data, n_sim) 
    # simulate trial n_sim times 
    preds = map(_->simulate_trail(actr, stimulus), 1:n_sim)
    LL = 0.0
    # compute approximate log likelihood for each trial in data 
    for d in data 
        p = max(mean(preds .== d.resp), 1/n_sim)
        # multiply log likelihood by number of replicates N
        LL += log(p)*d.N
    end
    return LL
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