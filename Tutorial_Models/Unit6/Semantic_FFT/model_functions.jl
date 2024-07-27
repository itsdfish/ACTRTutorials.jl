import Distributions: pdf, rand, logpdf

struct LNRC{T1, T2, T3} <: ContinuousUnivariateDistribution
    μ::T1
    σ::T2
    ϕ::T3
    c::Int
end

Broadcast.broadcastable(x::LNRC) = Ref(x)

LNRC(; μ, σ, ϕ, c) = LNRC(μ, σ, ϕ, c)

function rand(dist::LNRC)
    @unpack μ, σ, ϕ, c = dist
    x = @. rand(LogNormal(μ, σ)) + ϕ
    rt, resp = findmin(x)
    return rt
end

function rand_sim(dists)
    total_rt = 0.0
    resps = fill(0, length(dists))
    for (i, d) in enumerate(dists)
        @unpack μ, σ, ϕ = d
        x = @. rand(LogNormal(μ, σ)) + ϕ
        rt, resp = findmin(x)
        total_rt += rt
        resps[i] = resp
    end
    return resps, total_rt
end

rand(dist::LNRC, N::Int) = [rand(dist) for i = 1:N]

function logpdf(d::T, t::Float64) where {T <: LNRC}
    @unpack μ, σ, ϕ, c = d
    LL = 0.0
    for (i, m) in enumerate(μ)
        if i == c
            LL += logpdf(LogNormal(m, σ), t - ϕ)
        else
            LL += logccdf(LogNormal(m, σ), t - ϕ)
        end
    end
    return LL
end

function logpdf(d::LNRC{T1, T2, Vector{T3}}, t::Float64) where {T1, T2, T3}
    @unpack μ, σ, ϕ, c = d
    LL = 0.0
    for (i, m) in enumerate(μ)
        if i == c
            LL += logpdf(LogNormal(m, σ), t - ϕ[i])
        else
            LL += logccdf(LogNormal(m, σ), t - ϕ[i])
        end
    end
    return LL
end

function pdf(d::T, t::Float64) where {T <: LNRC}
    @unpack μ, σ, ϕ, c = d
    density = 1.0
    for (i, m) in enumerate(μ)
        if i == c
            density *= pdf(LogNormal(m, σ), t - ϕ)
        else
            density *= (1 - cdf(LogNormal(m, σ), t - ϕ))
        end
    end
    return density
end
