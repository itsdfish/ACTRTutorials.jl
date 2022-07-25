#######################################################################################
#                                   Load Packages
#######################################################################################
# set the working directory to the directory in which this file is contained
cd(@__DIR__)
# load the package manager
using Pkg
# activate the project environment
Pkg.activate("../../../")
# load the required packages
using Turing, StatsPlots, Revise, ACTRModels
# load all model functions
include("Grouped_Recall_1.jl")
# seed random number generator
Random.seed!(78455)
#######################################################################################
#                                   Generate Data
#######################################################################################
# fixed parameters and settings
fixed_parms = (s = 0.15,τ = -0.5,noise = true,mmp = true,mmpFun = simFun)
# mismatch penalty parameter for partial matching
δ = 1.0
# the number of blocks
n_blocks = 5
Data = map(x -> simulate(;δ, fixed_parms...), 1:n_blocks)
#######################################################################################
#                                    Define Model
#######################################################################################
@model model(Data, fixed_parms) = begin
    δ ~ truncated(Normal(1, 1.0), 0, Inf)
    Data ~ Grouped(δ, fixed_parms)
end
#######################################################################################
#                                 Estimate Parameters
#######################################################################################
# number of samples retained after warmup
n_samples = 1000
# the number of warmup or adaption samples
n_adapt = 1000
# the number of MCMC chains
n_chains = 4
# settings for the sampler
specs = NUTS(n_adapt, 0.8)
# estimate the parameters
chain = sample(model(Data, fixed_parms), specs, MCMCThreads(), n_samples, n_chains, progress=true)
#######################################################################################
#                                         Plot
#######################################################################################
pyplot()
ch = group(chain, :δ)
p1 = plot(ch, xaxis=font(5), yaxis=font(5), seriestype=(:traceplot),
  grid=false, size=(250,100), titlefont=font(5))
p2 = plot(ch, xaxis=font(5), yaxis=font(5), seriestype=(:autocorplot),
  grid=false, size=(250,100), titlefont=font(5))
p3 = plot(ch, xaxis=font(5), yaxis=font(5), seriestype=(:mixeddensity),
  grid=false, size=(250,100), titlefont=font(5))
pc = plot(p1, p2, p3, layout=(3,1), size=(200,200))
#######################################################################################
#                                  Posterior Predictive
#######################################################################################
preds = posterior_predictive(x -> simulate(;parms..., x...), chain, 1000, transpose_error)
p4 = histogram(preds, xaxis=font(7), yaxis=font(7), grid=false, size=(300,125),
    titlefont=font(7), leg=false, color=:grey, xlabel="Transposition Errors")