#######################################################################################
#                                   Load Packages
#######################################################################################
using Pkg
cd(@__DIR__)
Pkg.activate("../../..")
using Turing, StatsPlots, DataFrames, Revise, ACTRModels
include("Continuous_Time_PVT.jl")
Random.seed!(25425)
#######################################################################################
#                                   Generate Data
#######################################################################################
parms = (σ = 0.3,υ = 1.5,τ = 0.7)
n_trials = 50
rts = simulate(parms, n_trials)
#######################################################################################
#                                    Define Model
#######################################################################################
@model model(data) = begin
    υ ~ Normal(1.5, .5)
    τ = .7; σ = .3
    data ~ Markov(;υ, τ, σ)
end
#######################################################################################
#                                 Estimate Parameters
#######################################################################################
# Settings of the NUTS sampler.
n_samples = 1000
δ = 0.85
n_adapt = 1000
n_chains = 4
specs = NUTS(n_adapt, δ)
# Start sampling.
chain = sample(model(rts), specs, MCMCThreads(), n_samples, n_chains, progress=true)
#######################################################################################
#                                         Plot
#######################################################################################
pyplot()
ch = group(chain, :υ)
p1 = plot(ch, xaxis=font(5), yaxis=font(5), seriestype=(:traceplot),
  grid=false, size=(250,100), titlefont=font(5))
p2 = plot(ch, xaxis=font(5), yaxis=font(5), seriestype=(:autocorplot),
  grid=false, size=(250,100), titlefont=font(5))
p3 = plot(ch, xaxis=font(5), yaxis=font(5), seriestype=(:mixeddensity),
  grid=false, size=(250,100), titlefont=font(5))
pc = plot(p1, p2, p3, layout=(3,1), size=(600,600))
