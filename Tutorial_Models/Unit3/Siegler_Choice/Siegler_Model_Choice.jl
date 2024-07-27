import Distributions: logpdf, loglikelihood

struct Siegler{T1, T2, T3, T4} <: ContinuousUnivariateDistribution
    δ::T1
    τ::T2
    s::T3
    parms::T4
end

Siegler(; δ, τ, s, parms) = Siegler(δ, τ, s, parms)

loglikelihood(d::Siegler, data::Array{<:NamedTuple, 1}) = logpdf(d, data)

function logpdf(d::Siegler, data::Array{<:NamedTuple, 1})
    LL = computeLL(d.parms, data; δ = d.δ, τ = d.τ, s = d.s)
    return LL
end

function simulate(stimuli, parms; args...)
    # populate memory with addition facts
    chunks = populate_memory()
    # set blc parameters for each chunk
    set_baselevels!(chunks)
    # add chunks to declarative memory
    memory = Declarative(; memory = chunks)
    # add declarative memory to ACTR object
    actr = ACTR(; declarative = memory, parms..., args...)
    # randomize the stimuli
    shuffle!(stimuli)
    N = length(stimuli)
    data = Array{NamedTuple, 1}(undef, N)
    for (i, s) in enumerate(stimuli)
        # retrieval probabilities θs given request s
        Θs, r_chunks = retrieval_probs(actr; s...)
        # select chunk index from multinomial distribution
        idx = sample(1:length(Θs), weights(Θs))
        if idx == length(Θs)
            # retrieval failure coded as -100
            data[i] = (s..., resp = -100)
        else
            # record answer based on retrieved chunk
            data[i] = (s..., resp = r_chunks[idx].slots.sum)
        end
    end
    return data
end

function sim_fun(actr, chunk; request...)
    slots = chunk.slots
    p = 0.0
    δ = actr.parms.δ
    for (c, v) in request
        p += 0.1 * δ * abs(slots[c] - v)
    end
    return p
end

function populate_memory(act = 0.0)
    chunks = [
        Chunk(; num1 = num1, num2 = num2,
            sum = num1 + num2, act = act) for num1 = 0:5
        for num2 = 0:5
    ]
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
    chunks = populate_memory(zero(type))
    set_baselevels!(chunks)
    memory = Declarative(; memory = chunks)
    actr = ACTR(; declarative = memory, parms..., δ, τ, s)
    LL = 0.0
    for d in data
        if d.resp == -100
            # retrieval failure probability
            _, rf = retrieval_prob(actr, chunks[1]; num1 = d.num1, num2 = d.num2)
            LL += log(rf) * d.N
        else
            # probability of retrieving a chunk with sum = resp
            t_chunks = get_chunks(actr; sum = d.resp)
            θᵣ, _ = retrieval_prob(actr, t_chunks; num1 = d.num1, num2 = d.num2)
            LL += log(θᵣ) * d.N
        end
    end
    return LL
end

function response_histogram(df, stimuli; kwargs...)
    vals = NamedTuple[]
    for (num1, num2) in stimuli
        idx = @. ((df[:, :num1] == num1) & (df[:, :num2] == num2)) |
           ((df[:, :num1] == num2) & (df[:, :num2] == num1))
        subdf = df[idx, :]
        str = string(num1, "+", num2)
        v = filter(x -> x != -100, subdf[:, :resp])
        push!(vals, (title = str, data = v))
    end
    p = histogram(layout = (2, 3), leg = false, xlims = (0, 10), xlabel = "Response",
        ylabel = "Proportion",
        size = (600, 600), xaxis = font(12), yaxis = font(12), titlefont = font(12),
        grid = false; kwargs...)
    for (i, v) in enumerate(vals)
        histogram!(p, v.data, subplot = i, title = v.title, bar_width = 1, color = :grey,
            grid = false,
            normalize = :probability, ylims = (0, 1))
    end
    return p
end
