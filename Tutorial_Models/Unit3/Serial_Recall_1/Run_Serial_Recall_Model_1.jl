#######################################################################################
#                                   Load Packages
#######################################################################################
cd(@__DIR__)
using Pkg
Pkg.activate("../../..")
using Turing, StatsPlots, Revise, ACTRModels
include("Serial_Recall_Model_1.jl")
Random.seed!(2920)
#######################################################################################
#                                   Generate Data
#######################################################################################
# mismatch penalty parameter
δ = 1.0
# number of blocks
n_blocks = 10
# number of items per block
n_items = 10
# fixed parameters
parms = (s = 0.3,τ=-100.0,mmp = true,noise = true,mmpFun = penalty)
data = map(_ -> simulate(parms, n_items; δ), 1:n_blocks);
#######################################################################################
#                                    Define Model
#######################################################################################
@model model(data, parms, n_items) = begin
    δ ~ truncated(Normal(1.0, 0.5), 0, Inf)
    data ~ SerialRecall(δ, parms, n_items)
end
#######################################################################################
#                                 Estimate Parameters
#######################################################################################
# Settings of the NUTS sampler.
n_samples = 1000
n_adapt = 1000
n_chains = 4
specs = NUTS(n_adapt, 0.8)
# Start sampling.
chain = sample(model(data, parms, n_items), specs, MCMCThreads(), n_samples, n_chains, progress=true)
#######################################################################################
#                                         Plot
#######################################################################################
pyplot()
font_size = 12
ch = group(chain, :δ)
p1 = plot(ch, xaxis=font(font_size), yaxis=font(font_size), seriestype=(:traceplot),
  grid=false, size=(250,100), titlefont=font(font_size))
p2 = plot(ch, xaxis=font(font_size), yaxis=font(font_size), seriestype=(:autocorplot),
  grid=false, size=(250,100), titlefont=font(font_size))
p3 = plot(ch, xaxis=font(font_size), yaxis=font(font_size), seriestype=(:mixeddensity),
  grid=false, size=(250,100), titlefont=font(font_size))
pcτ = plot(p1, p2, p3, layout=(3,1), size=(800,600))
#######################################################################################
#                                  Posterior Predictive
#######################################################################################
preds = posterior_predictive(x -> transpostions(parms, n_items; x...), chain, 100)
preds = vcat(preds...)
p4 = histogram(preds, xlabel="Position Difference", ylabel="Retrieval Probability", xaxis=font(font_size),
    yaxis=font(font_size), grid=false, normalize = :probability, color=:grey, leg=false, size=(800,400),
    titlefont=font(font_size), linewidth=.3)