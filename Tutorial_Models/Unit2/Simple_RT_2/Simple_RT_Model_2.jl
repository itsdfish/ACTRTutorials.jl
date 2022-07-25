import Distributions: logpdf, rand, loglikelihood

struct RT{T1,T2,T3} <: ContinuousUnivariateDistribution
    blc::T1
    τ::T2
    parms::T3
end

RT(;blc, τ, parms) = RT(blc, τ, parms)

loglikelihood(d::RT, data::Array{<:NamedTuple,1}) = logpdf(d, data)

function logpdf(d::RT, data::Array{<:NamedTuple,1})
    LL = computeLL(d.blc, d.τ, d.parms, data)
    return LL
end

function simulate(parms; blc, τ)
    # Create chunk
    chunks = [Chunk()]
    # add chunk to declarative memory
    memory = Declarative(;memory=chunks)
    # create ACTR object and pass parameters
    actr = ACTR(;declarative=memory, parms..., blc, τ)
    # retrieve chunk
    chunk = retrieve(actr)
    # 2 if empty, 1 otherwise
    resp = isempty(chunk) ? resp = 2 : 1
    # compute reaction time 
    rt = compute_RT(actr, chunk) + actr.parms.ter
    return (resp = resp,rt = rt)
end

function computeLL(blc, τ, parms, data)
    (;s,ter) = parms
    LL = 0.0
    σ = s * pi / sqrt(3)
    # define distribution object
    dist = LNR(;μ=-[blc,τ], σ, ϕ=ter)
    # compute log likelihood for each data point
    for d in data
        LL += logpdf(dist, d...)
    end
    return LL
end
