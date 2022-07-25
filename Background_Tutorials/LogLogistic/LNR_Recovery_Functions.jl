import Distributions: logpdf, loglikelihood

struct LNR′{T1,T2} <: ContinuousUnivariateDistribution
    μ::T1
    s::T2
end

loglikelihood(d::LNR′, data::Array{<:Tuple,1}) = logpdf(d, data)

function logpdf(d::LNR′, data::Array{<:Tuple,1})
    σ = d.s*pi/sqrt(3)
    LL = 0.0
    for v in data
        LL += logpdf(LNR(μ=-d.μ, σ=σ, ϕ=0.0), v)
    end
    return LL
end

@model model(data, Nr) = begin
    μ = Vector{Real}(undef,Nr)
    μ .~ Normal(0, 1)
    s ~ Uniform(0, pi/2)
    z = tan(s)
    data ~ LNR′(μ, z)
end

function _loglogistic(μ, s, deterministic)
    x = @. exp(rand(Logistic(-μ, s)))
    x[end] = deterministic ? exp(-μ[end]) : x[end]
    rt,resp = findmin(x)
    return resp,rt
end

function loglogistic(μ, s, N, deterministic)
    return [_loglogistic(μ, s, deterministic) for i in 1:N]
end

function run_sim(model, μt, st, Nr, N, deterministic)
    Nreps = length(μt)
    Nchains = 2
    i = 0
    col_names = [Symbol("eμ",i) for i in 1:Nr]
    push!(col_names, :es, :rhat, :r_failures)
    results = DataFrame(fill(0.0, 0, Nr+3), col_names)
    chain = Chains(rand(10))
    data = loglogistic([1], 1, 1, deterministic)
    for (m,s) in zip(μt, st)
        i += 1
        running = true
        while running 
            println("Iteration ", i, " of ", Nreps)
            println()
            data = loglogistic(m, s, N, deterministic)
            n_samples = 1000
            n_adapt = 1000
            specs = NUTS(n_adapt, .65)
            n_chains = 4
            run_sampler() = sample(model(data, Nr), specs, MCMCThreads(), n_samples, n_chains, progress=true)
            result = run_with_timeout(60, run_sampler, 1)
            if result != :timeout 
                running = false
                chain = result[1]
            end
        end
        df = describe(chain)[1]
        means = df[:,:mean]
        rhat = maximum(df[:,:rhat])
        r_failures = mean(x->x[1] == Nr, data)
        println(df)
        temp = get(chain, :s)
        se = mean(tan.(temp.s))
        push!(results, [means[2:end]...  se  rhat r_failures])
    end
    return results
end

using Distributed
function run_with_timeout(timeout, f::Function, wid::Int)
    result = RemoteChannel(()->Channel{Tuple}(1));
    @spawnat wid put!(result, (f(),myid()))
    res = :timeout
    time_elapsed = 0.0
    while time_elapsed < timeout && !isready(result)
        sleep(0.5)
        time_elapsed += 0.5
    end
    if !isready(result)
        println("Timeout! at $wid")
    else
        res = take!(result)
    end
    return res
end