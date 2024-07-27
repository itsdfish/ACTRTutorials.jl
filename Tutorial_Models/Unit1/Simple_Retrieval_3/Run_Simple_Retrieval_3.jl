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
include("Simple_Retrieval_3.jl")
include("../../../Utilities/Utilities.jl")
# seed random number generator
Random.seed!(65185);
#######################################################################################
#                                   Generate Data
#######################################################################################
# Number of trials
n_trials = 50
# number of items in stimulus list
n_items = 10
# Sample stimulis
stimuli = sample_stimuli(n_items, n_trials)
# Retrieval threshold parameter
τ = 0.5
# Mismatch penalty parameter
δ = 1.0
# Fixed parameters
parms = (blc = 1.5, s = 0.2, mmp = true)
# Simulate model
temp = simulate(n_items, stimuli, parms; δ, τ)
# Tabulate counts of unique responses
data = unique_data(temp)
#######################################################################################
#                                    Define Model
#######################################################################################
@model model(data, parms, n_items) = begin
    δ ~ Normal(1.0, 0.5)
    τ ~ Normal(0.5, 0.5)
    data ~ Retrieval(τ, δ, n_items, parms)
end
#######################################################################################
#                                 Estimate Parameters
#######################################################################################
# Settings of the NUTS sampler.
n_samples = 1000
delta = 0.85
n_adapt = 1000
n_chains = 4
specs = NUTS(n_adapt, delta)
# Start sampling.
chain = sample(
    model(data, parms, n_items),
    specs,
    MCMCThreads(),
    n_samples,
    n_chains,
    progress = true
)
#######################################################################################
#                                         Plot
#######################################################################################
pyplot()
font_size = 12
ch = group(chain, :τ)
p1 = plot(ch, xaxis = font(font_size), yaxis = font(font_size), seriestype = (:traceplot),
    grid = false, size = (250, 100), titlefont = font(font_size))
p2 =
    plot(ch, xaxis = font(font_size), yaxis = font(font_size), seriestype = (:autocorplot),
        grid = false, size = (250, 100), titlefont = font(font_size))
p3 =
    plot(ch, xaxis = font(font_size), yaxis = font(font_size), seriestype = (:mixeddensity),
        grid = false, size = (250, 100), titlefont = font(font_size))
pcτ = plot(p1, p2, p3, layout = (3, 1), size = (600, 600))

pyplot()
font_size = 12
ch = group(chain, :δ)
p1 = plot(ch, xaxis = font(font_size), yaxis = font(font_size), seriestype = (:traceplot),
    grid = false, size = (250, 100), titlefont = font(font_size))
p2 =
    plot(ch, xaxis = font(font_size), yaxis = font(font_size), seriestype = (:autocorplot),
        grid = false, size = (250, 100), titlefont = font(font_size))
p3 =
    plot(ch, xaxis = font(font_size), yaxis = font(font_size), seriestype = (:mixeddensity),
        grid = false, size = (250, 100), titlefont = font(font_size))
pcδ = plot(p1, p2, p3, layout = (3, 1), size = (600, 600))
#######################################################################################
#                                  Posterior Predictive
#######################################################################################
preds = posterior_predictive(x -> simulate(n_items, stimuli, parms; x...), chain, 1000)
get_counts(data, v) = count(x -> x.matches == v, data)
counts_correct = get_counts.(preds, true)
p4 = histogram(counts_correct, xlabel = "Number Correct", ylabel = "Density",
    xaxis = font(12), yaxis = font(12),
    grid = false, norm = true, color = :grey, leg = false, size = (800, 400),
    titlefont = font(12),
    bar_width = 1)

counts_incorrect = get_counts.(preds, false)
p4 = histogram(counts_incorrect, xlabel = "Number Incorrect", ylabel = "Density",
    xaxis = font(12), yaxis = font(12),
    grid = false, norm = true, color = :grey, leg = false, size = (800, 400),
    titlefont = font(12),
    bar_width = 1)
