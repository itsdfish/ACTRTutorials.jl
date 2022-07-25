#######################################################################################
#                                   Load Packages
#######################################################################################
# change directory to this files containing folder
cd(@__DIR__)
# import package manager
using Pkg
# activate project environment
Pkg.activate("../../..")
# import required packages
using Turing, StatsPlots, Revise, ACTRModels, Distributions
# import required model functions
include("Simple_RT_Model_4.jl")
# initialize random number generator
Random.seed!(3401);
#######################################################################################
#                                   Generate Data
#######################################################################################
# Number of trials
n_trials = 50
# Number of unique stimuli
n_items = 10
# Sample stimuli
stimuli = sample_stimuli(n_items, n_trials)
# Mismatch penalty parameter
δ = 1.0
# Logistic scale parameter for activation noise
s = 0.3
# Retrieval Threshold parameter
τ = 0.5
# Perceptual-motor and conflict resolution time
ter = (0.05 + 0.085 + 0.05) + (0.05 + 0.06)
# Fixed Parameters
parms = (noise = true,ter = ter, blc=1.25, mmp=true)
# Generate Data
data = simulate(n_items, stimuli, parms; δ, τ, s)
#######################################################################################
#                                    Define Model
#######################################################################################
@model model(data, parms, n_items) = begin
    δ ~ Normal(1.0, 0.5)
    s ~ Truncated(Normal(0.3, 0.5), 0.0, Inf)
    τ ~ Normal(0.5, 0.5)
    data ~ RT(δ, s, τ, n_items, parms)
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
chain = sample(model(data, parms, n_items), specs, MCMCThreads(), n_samples, n_chains, progress=true)
#######################################################################################
#                                      Summarize
#######################################################################################
println(chain)
#######################################################################################
#                                         Plot
#######################################################################################
pyplot()
ch = group(chain, :δ)
font_size = 12
p1 = plot(ch, xaxis=font(font_size), yaxis=font(font_size), seriestype=(:traceplot),
  grid=false, size=(250,100), titlefont=font(font_size))
p2 = plot(ch, xaxis=font(font_size), yaxis=font(font_size), seriestype=(:autocorplot),
  grid=false, size=(250,100), titlefont=font(font_size))
p3 = plot(ch, xaxis=font(font_size), yaxis=font(font_size), seriestype=(:mixeddensity),
  grid=false, size=(250,100), titlefont=font(font_size))
pcblc = plot(p1, p2, p3, layout=(3,1), size=(600,600))

ch = group(chain, :s)
p1 = plot(ch, xaxis=font(font_size), yaxis=font(font_size), seriestype=(:traceplot),
  grid=false, size=(250,100), titlefont=font(font_size))
p2 = plot(ch, xaxis=font(font_size), yaxis=font(font_size), seriestype=(:autocorplot),
  grid=false, size=(250,100), titlefont=font(font_size))
p3 = plot(ch, xaxis=font(font_size), yaxis=font(font_size), seriestype=(:mixeddensity),
  grid=false, size=(250,100), titlefont=font(font_size))
pcs = plot(p1, p2, p3, layout=(3,1), size=(600,600))
#######################################################################################
#                                  Posterior Predictive
#######################################################################################
temp = posterior_predictive(x -> simulate(n_items, stimuli, parms,; x...), chain, 1000)
preds = vcat(temp...)
correct = filter(x-> x.correct == 1, preds)
rts_correct = map(x->x.rt, correct)
p_correct = mean(x->x.correct == 1, preds)
correct_dist = histogram(rts_correct, xlabel="RT", ylabel="Density", xaxis=font(font_size), yaxis=font(font_size),
    grid=false, norm=true, color=:grey, leg=false, size=(600,300), title="Correct", titlefont=font(font_size),
    xlims=(0,1.5))
correct_dist[1][1][:y] *= p_correct

incorrect = filter(x-> x.correct == 2, preds)
rts_incorrect = map(x->x.rt, incorrect)
incorrect_dist = histogram(rts_incorrect, xlabel="RT", ylabel="Density", xaxis=font(font_size), yaxis=font(font_size),
    grid=false, norm=true, color=:grey, leg=false, size=(600,300), title="Incorrect", titlefont=font(font_size),
    xlims=(0,1.5))
incorrect_dist[1][1][:y] *= (1 - p_correct)
plot(correct_dist, incorrect_dist, layout=(2,1), ylims=(0,3), size=(800, 500))
