using Parameters, StatsBase, Random
import Distributions: logpdf, rand, loglikelihood

struct Siegler{T1,T2,T3,T4} <: ContinuousUnivariateDistribution
    δ::T1
    τ::T2
    s::T3
    parms::T4
end

Siegler(;δ, τ, s, parms) = Siegler(δ, τ, s, parms)

loglikelihood(d::Siegler, data::Array{<:NamedTuple,1}) = logpdf(d, data)

function logpdf(d::Siegler, data::Array{<:NamedTuple,1})
    LL = computeLL(d.parms, data; δ=d.δ, τ=d.τ, s=d.s)
    return LL
end

function simulate(stimuli, parms; args...)
    # populate chunks
    chunks = populate_memory()
    # set the base level constant based on sum slot
    set_baselevels!(chunks)
    # create a declarative memory object
    memory = Declarative(;memory=chunks)
    # create an ACTR model object
    actr = ACTR(;declarative=memory, parms..., args...)
    shuffle!(stimuli)
    N = length(stimuli)
    data = Array{NamedTuple,1}(undef, N)
    # loop through trials
    for (i,s) in enumerate(stimuli)
        # retrieve a chunk based on retrieval request s
        chunk = retrieve(actr; s...)
        # compute reaction time 
        rt = compute_RT(actr, chunk) + parms.ter
        if isempty(chunk)
            # retrieval failure
            data[i] = (s...,resp = -100,rt = rt)
        else
            # chunk retrieved. 
            data[i] = (s...,resp = chunk[1].slots.sum,rt = rt)
        end
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

function populate_memory(act=0.0)
    chunks = [Chunk(;num1=num1,num2=num2,
        sum=num1 + num2,act=act) for num1 in 0:5
        for num2 in 0:5]
    pop!(chunks)
    return chunks
end

function set_baselevels!(chunks)
    for chunk in chunks
        if chunk.slots.sum < 5
            chunk.bl = 0.65
        end
    end
    return nothing
end

function computeLL(parms, data; δ, τ, s)
    type = typeof(δ)
    # populate chunks
    chunks = populate_memory(zero(δ))
    # set base level constant based on sum slot
    set_baselevels!(chunks)
    # create declarative memory object
    memory = Declarative(;memory=chunks)
    # create ACTR model object
    actr = ACTR(;declarative=memory, parms..., δ, τ, s)
    # remove random values from activation
    actr.parms.noise = false
    N = length(chunks) + 1
    @unpack s,ter,τ = actr.parms
    LL = 0.0; idx = 0; μ = Array{type,1}(undef, N)
    σ = s * pi / sqrt(3)
    ϕ = ter
    for (num1,num2,resp,rt) in data
        # compute activation
        compute_activation!(actr; num1, num2)
        # extract mean activation values
        map!(x -> x.act, μ, chunks)
        # last mean activation is for retrieval failure
        μ[end] = τ
        # log normal race distribution 
        dist = LNR(;μ=-μ, σ, ϕ)
        # no retrieval error
        if resp != -100 
            # get all chunk indices such that sum = response
            indices = find_indices(actr; sum=resp)
            log_probs = zeros(type, length(indices))
            # loop over each component of mixture and compute log likelihood
            for (c,idx) in enumerate(indices)
                log_probs[c] = logpdf(dist, idx, rt)
            end
            # compute marginal likelihood
            LL += logsumexp(log_probs)
        else
            # retrieval failure 
            LL += logpdf(dist, N, rt)
        end
    end
    return LL
end

function rt_histogram(df, stimuli; kwargs...)
    vals = NamedTuple[]
    for (num1,num2) in stimuli
        idx = @. ((df[:,:num1] == num1 ) & (df[:,:num2] == num2)) | ((df[:,:num1] == num2) & (df[:,:num2] == num1))
        subdf = df[idx,:]
        str = string(num1, "+", num2)
        temp = filter(x -> x[:resp] != -100, subdf)
        g = groupby(temp, :resp)
        rt_resp = combine(g, :rt => mean)
        push!(vals, (title = str, data = rt_resp))
    end
    p = bar(layout=(2,3), leg=false, xlims=(0,10), xlabel="Response", ylabel="Mean RT",
        size=(600,600), xaxis=font(6), yaxis=font(6), titlefont=font(7), grid=false; kwargs...)
    for (i,v) in enumerate(vals)
        @df v.data bar!(p, :resp, :rt_mean, subplot=i, title=v.title, bar_width=1, color=:grey,
            grid=false, ylims=(0,4.5), xlims=(-.5,9))
    end
    return p
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
        size=(600,600), xaxis=font(6), yaxis=font(6), titlefont=font(7), grid=false; kwargs...)
    for (i,v) in enumerate(vals)
        histogram!(p, v.data, subplot=i, title=v.title, bar_width=1, color=:grey, grid=false,
        normalize=:probability, ylims=(0,1))
    end
    return p
end

function parse_lisp_data(data)
    new_data = NamedTuple[]
    for row in eachrow(data)
        temp = (num1 = row[1],num2 = row[2],
        resp = numeric_response(row[3]),rt = row[4])
        push!(new_data, temp)
    end
    return vcat(new_data)
end

function numeric_response(resp)
    if resp == "zero"
        return 0
    elseif resp == "one"
        return 1
    elseif resp == "two"
        return 2
    elseif resp == "three"
        return 3
    elseif resp == "four"
        return 4
    elseif resp == "five"
        return 5
    elseif resp == "six"
        return 6
    elseif resp == "seven"
        return 7
    elseif resp == "eight"
        return 8
    elseif resp == "nine"
        return 9
    elseif resp == "nil"
        return -100
    end
end