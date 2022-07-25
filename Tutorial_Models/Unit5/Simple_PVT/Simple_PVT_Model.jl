using Parameters, StatsBase, Distributions
import Distributions: logpdf, loglikelihood

struct SimplePVT{T1,T2,T3} <: ContinuousUnivariateDistribution
    υ::T1
    λ::T2
    parms::T3
end

loglikelihood(d::SimplePVT, data::Array{Float64,1}) = logpdf(d, data)

SimplePVT(;υ, parms) = SimplePVT(υ, parms)

function logpdf(d::SimplePVT, data::Array{Float64,1})
    LL = pvt_log_like(data; υ=d.υ,λ=d.λ, d.parms...)
    return LL
end

function show_cycles(n_cycles)
    output = []
    non_att ="CR(0) "
    for i in 1:(n_cycles-1)
        non_att = non_att*string("CR(", i, ") ")
    end
    non_att = non_att*string("R(", n_cycles-1, ")")
    push!(output, non_att)
    for n_lapses in 0:(n_cycles-2)
        temp = "CR(0) "
        for i in 1:(n_lapses)
            temp = temp*string("CR(", i, ") ")
        end
        temp = temp*string("A(", n_lapses, ") ")

        for i in n_lapses:(n_cycles-2)
            temp = temp*string("CR(", i, ") ")
        end
        temp = temp*string("R(", n_cycles-2, ")")
        push!(output, temp)
    end
    return output
end
