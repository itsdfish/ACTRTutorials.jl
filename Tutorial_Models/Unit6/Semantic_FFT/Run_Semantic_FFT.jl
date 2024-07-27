#######################################################################################
#                                   Load Packages
#######################################################################################
# change current directory to file's directory
cd(@__DIR__)
# load package manager
using Pkg
# activate the package environment
Pkg.activate("../../..")
# load packages and model functions
using Revise, StatsPlots, ACTRModels, Distributions, FFTDists, DifferentialEvolutionMCMC
using MCMCChains
include("Semantic_FFT_Model.jl")
include("model_functions.jl")
Random.seed!(37101)
#######################################################################################
#                                   Generate Data
#######################################################################################
# true value for base level constant
blc = 1.5
# true value for mismatch penalty
δ = 1.0
# fixed parameters
fixed_parms = (noise = true, τ = 0.0, s = 0.2, mmp = true)
stimuli = get_stimuli()
# repeat simulation 10 times per stimulus
n_reps = 10
data = map(s -> simulate(fixed_parms, s, n_reps; blc, δ), stimuli)
#######################################################################################
#                                    Define Model
#######################################################################################
priors = (
    blc = (Normal(1.5, 1),),
    δ = (Truncated(Normal(1.0, 0.5), 0.0, Inf),)
)
bounds = ((-Inf, Inf), (eps(), Inf))
#######################################################################################
#                                 Estimate Parameters
#######################################################################################
model = DEModel(fixed_parms; priors, model = loglike, data)
de = DE(; bounds, burnin = 1000, priors, n_groups = 2, Np = 4)
n_iter = 2000
chains = sample(model, de, MCMCThreads(), n_iter, progress = true)
#######################################################################################
#                                         Plot
#######################################################################################
pyplot()
posteriors = plot(chains, seriestype = :pooleddensity, grid = false, titlefont = font(10),
    xaxis = font(8), yaxis = font(8), color = :grey, size = (300, 250))
#######################################################################################
#                                  Posterior Predictive
#######################################################################################
rt_preds(s) =
    posterior_predictive(x -> simulate(fixed_parms, s, n_reps; x...), chains, 1000)
temp_preds = map(s -> rt_preds(s), stimuli)
preds = merge.(temp_preds)
grid_plot(preds, stimuli)
