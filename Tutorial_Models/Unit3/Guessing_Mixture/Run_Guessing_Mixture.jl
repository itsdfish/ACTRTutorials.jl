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
using Turing, StatsPlots, ACTRModels
# load all model functions
include("Guessing_Mixture.jl")
# seed random number generator
Random.seed!(44301);
#######################################################################################
#                                   Generate Data
#######################################################################################
# number of trials for targets and foils
n_trials = (t = 80,f = 20)
# retrieval threshold parameter
τ = 0.5
# guessing parameter
θg = 0.8
# fixed parameters for base level constant and logistic scale
fixed_parms = (blc = 1.5,s = 0.4)
# generate data
data = simulate(fixed_parms, n_trials; τ, θg)
#######################################################################################
#                                    Define Model
#######################################################################################
@model model(data, fixed_parms) = begin
    τ ~ Normal(0.5, 0.5)
    θg ~ Beta(8, 2)
    data ~ Retrieval(τ, θg, fixed_parms)
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
chain = sample(model(data, fixed_parms), specs, MCMCThreads(), n_samples, n_chains, progress=true)
#######################################################################################
#                                         Plot
#######################################################################################
pyplot()
chτ = group(chain, :τ)
font_size = 12
p1 = plot(chτ, xaxis=font(font_size), yaxis=font(font_size), seriestype=(:traceplot),
  grid=false, size=(250,100), titlefont=font(font_size))
p2 = plot(chτ, xaxis=font(font_size), yaxis=font(font_size), seriestype=(:autocorplot),
  grid=false, size=(250,100), titlefont=font(font_size))
p3 = plot(chτ, xaxis=font(font_size), yaxis=font(font_size), seriestype=(:mixeddensity),
  grid=false, size=(250,100), titlefont=font(font_size))
pcτ = plot(p1, p2, p3, layout=(3,1), size=(600,600))

chθg = group(chain, :θg)
p1 = plot(chθg, xaxis=font(font_size), yaxis=font(font_size), seriestype=(:traceplot),
  grid=false, size=(250,100), titlefont=font(font_size))
p2 = plot(chθg, xaxis=font(font_size), yaxis=font(font_size), seriestype=(:autocorplot),
  grid=false, size=(250,100), titlefont=font(font_size))
p3 = plot(chθg, xaxis=font(font_size), yaxis=font(font_size), seriestype=(:mixeddensity),
  grid=false, size=(250,100), titlefont=font(font_size))
pcθg = plot(p1, p2, p3, layout=(3,1), size=(600,600))
#######################################################################################
#                                  Posterior Predictive
#######################################################################################
preds = posterior_predictive(x -> simulate(fixed_parms, n_trials; x...), chain, 1000)
target = map(x -> x.target, preds)
foil = map(x -> x.foil, preds)
p4 = histogram(target, xlabel="Data", ylabel="Frequency", xaxis=font(font_size), yaxis=font(font_size),
 grid=false, color=:grey, leg=false, size=(600,300), bar_width=1, title="Target",
    titlefont=font(font_size))
p5 = histogram(foil, xlabel="Data",ylabel="Frequency", xaxis=font(font_size), yaxis=font(font_size),
  grid=false, color=:grey, leg=false, size=(600,300), bar_width=1, title="Foil",
  titlefont=font(font_size))
