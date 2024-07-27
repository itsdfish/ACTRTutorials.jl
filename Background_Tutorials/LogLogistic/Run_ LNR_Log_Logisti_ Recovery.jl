cd(@__DIR__)
using Pkg
Pkg.activate("../../")
using Turing, StatsPlots, Distributions, ACTRModels, GLM, Random, CSV, DataFrames
using MCMCChains
include("LNR_Recovery_Functions.jl")
##############################################################################
#                               Set Parameters
##############################################################################
Random.seed!(11885)
N = 100
n_sim = 500
Nr = 3
#threshold/response deadline
deterministic = true
μt = [rand(Uniform(-1.5, 1.5), Nr) for i = 1:n_sim]
st = rand(Uniform(0.3, 1), n_sim)
parms = [Symbol("tμ", i) for i = 1:Nr]
true_parms = DataFrame(hcat(μt...)', parms)
true_parms[:, :ts] = st
estimates = run_sim(model, μt, st, Nr, N, deterministic)
results = [true_parms estimates]
filter!(x -> x[:rhat] < 1.05, results)
##############################################################################
#                               Plot Results
##############################################################################
pyplot()
μ = -1:0.2:1
ss = 0.3:0.2:1
plot(layout = (Nr + 1), size = (800, 500), xaxis = font(7), yaxis = font(7))
linear(β0, β1, x) = β0 .+ β1 * x
μta = hcat(μt...)'
coefs = DataFrame(
    parm = Symbol[],
    intercept = Float64[],
    slope = Float64[],
    correlation = Float64[]
)
parms = [Symbol("μ", i) for i = 1:Nr]
push!(parms, :s)
for (i, parm) in enumerate(parms)
    θt = Symbol("t", parm)
    θe = Symbol("e", parm)
    lb = minimum([results[:, θt]; results[:, θe]]) * 1.1
    ub = maximum([results[:, θt]; results[:, θe]]) * 1.1
    @df results scatter!(cols(θt), cols(θe), xlabel = string(parm, " true"),
        ylabel = string(parm, " estimate"), grid = false,
        color = :grey, leg = false, subplot = i, markersize = 3.5, markerstrokewidth = 1,
        xlims = (lb, ub), ylims = (lb, ub),
        xaxis = font(14), yaxis = font(14))
    ols = lm(@eval(@formula($θe ~ $θt)), results)
    βs = coef(ols)
    ρ = cor(results[!, θe], results[!, θt])
    push!(coefs, [parm, βs..., ρ])
    plot!(μ, linear(βs..., μ), color = :darkred, line = :dash, linewidth = 2, subplot = i)
end
deterministic ? deadline = "deterministic" : deadline = "stochastic"
savefig(string("Recovery_LogLogistic_LNR_", deadline, "_", Nr, ".png"))
CSV.write(string("Recovery_LogLogistic_Coefs_", deadline, "_", Nr, ".csv"), coefs)
CSV.write(string("Recovery_LogLogistic_All_Results_", deadline, "_", Nr, ".csv"), results)
