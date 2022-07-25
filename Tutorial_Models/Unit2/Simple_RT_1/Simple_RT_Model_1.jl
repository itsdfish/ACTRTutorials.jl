import Distributions: logpdf, rand, loglikelihood

struct RT{T1,T2} <: ContinuousUnivariateDistribution
    blc::T1
    parms::T2
end

RT(; blc, parms) = RT(blc, parms)

loglikelihood(d::RT, data::Array{Float64,1}) = logpdf(d, data)

function logpdf(d::RT, rts::Array{Float64,1})
    LL = computeLL(rts; blc=d.blc, d.parms...)
    return LL
end

function simulate(parms; blc)
    # create a chunk
    chunks = [Chunk()]
    # add the chunk to declarative memory
    memory = Declarative(;memory=chunks)
    # create the ACTR object and pass parameters
    actr = ACTR(;declarative=memory, parms..., blc)
    # retrieve the chunk
    chunk = retrieve(actr)
    # compute the reaction time 
    rt = compute_RT(actr, chunk) + actr.parms.ter
    return rt
end

function computeLL(rts; blc, ter, s, kwargs...)
    # define standard deviation of activation noise
    σ = s * pi / sqrt(3)
    # Define the lognormal distribution
    dist = LNR(;μ=[-blc], σ, ϕ=ter)
    # sum the log likelihood of all rts
    return sum(logpdf.(dist, 1, rts))
end