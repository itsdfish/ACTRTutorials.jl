#######################################################################################
#                                   Load Packages
#######################################################################################
# change workding directory to the directory of this file
cd(@__DIR__)
# import package manager
using Pkg
# activate project environment
Pkg.activate("../../..")
# import required packages
using Turing, StatsPlots, Revise, VisualSearchACTR, Distributions
# import required model functions
include("Visual_Search_Model.jl")
# initialize random number generator
Random.seed!(50998);
#######################################################################################
#                                   Generate Data
#######################################################################################
# number of trials
n_trials = 10
topdown_weight = 0.66
stimuli,all_fixations = simulate(;n_trials, topdown_weight);
#######################################################################################
#                                    Define Model
#######################################################################################
@model function model(stimuli, all_fixations)
    topdown_weight ~ Normal(0.66, 0.5)
    all_fixations ~ VisualSearch(topdown_weight, stimuli)
end
n_samples = 1000
delta = 0.85
n_adapt = 1000
n_chains = 4
specs = NUTS(n_adapt, delta)
# Start sampling.
@time chain = sample(model(stimuli, all_fixations), specs, MCMCThreads(), n_samples, n_chains, progress=true)
#######################################################################################
#                                         Plot
#######################################################################################
pyplot()
ch = group(chain, :topdown_weight)
font_size = 12
p1 = plot(ch, xaxis=font(font_size), yaxis=font(font_size), seriestype=(:traceplot),
  grid=false, size=(250,100), titlefont=font(font_size))
p2 = plot(ch, xaxis=font(font_size), yaxis=font(font_size), seriestype=(:autocorplot),
  grid=false, size=(250,100), titlefont=font(font_size))
p3 = plot(ch, xaxis=font(font_size), yaxis=font(font_size), seriestype=(:mixeddensity),
  grid=false, size=(250,100), titlefont=font(font_size))
pcblc = plot(p1, p2, p3, layout=(3,1), size=(600,600))
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