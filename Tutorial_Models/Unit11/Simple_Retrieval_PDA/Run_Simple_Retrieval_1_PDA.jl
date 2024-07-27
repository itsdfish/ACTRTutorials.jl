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
using DifferentialEvolutionMCMC, StatsPlots, Revise, ACTRModels, MCMCChains
# load all model functions
include("Simple_Retrieval_1_PDA.jl")
# seed random number generator
Random.seed!(2050);
#######################################################################################
#                                   Generate Data
#######################################################################################
# The number of trials
n_trials = 50
# True value of retrieval threshold
τ = 0.5
# Fixed parameters
parms = (blc = 1.5, s = 0.4)
# Simulate the number of correct retrievals
data = simulate(parms, n_trials; τ)
#######################################################################################
#                                    Define Model
#######################################################################################
# prior on τ
priors = (
    τ = (Normal(0.5, 0.5),),
)
# lower and upper bounds for τ
bounds = ((-Inf, Inf),)

# generate model object
model = DEModel(; priors, model = loglike, data, n_trials, parms)

# generate sampler object
de = DE(; bounds, burnin = 1000, priors)
n_iter = 2000
# perform parameter estimation 
chain = sample(model, de, MCMCThreads(), n_iter, progress = true)
#######################################################################################
#                                         Plot
#######################################################################################
pyplot()
ch = group(chain, :τ)
p1 = plot(ch, xaxis = font(10), yaxis = font(10), seriestype = (:traceplot),
    grid = false, size = (250, 100), titlefont = font(10))
p2 = plot(ch, xaxis = font(10), yaxis = font(10), seriestype = (:autocorplot),
    grid = false, size = (250, 100), titlefont = font(10))
p3 = plot(ch, xaxis = font(10), yaxis = font(10), seriestype = (:mixeddensity),
    grid = false, size = (250, 100), titlefont = font(10))
pc = plot(p1, p2, p3, layout = (3, 1), size = (600, 600))
#######################################################################################
#                                  Posterior Predictive
#######################################################################################
preds = posterior_predictive(x -> simulate(parms, n_trials; x...), chain, 1000)
p4 = histogram(preds, xlabel = "Number Retrieved", ylabel = "Density", xaxis = font(12),
    yaxis = font(12),
    grid = false, norm = true, color = :grey, leg = false, size = (800, 400),
    titlefont = font(12),
    bar_width = 1)
