#######################################################################################
#                                   Load Packages
#######################################################################################
# change directory of this file
cd(@__DIR__)
# pakcage manager
using Pkg
# activate this environment of this project
Pkg.activate("../../..")
# load dependencies
using Turing, DataFrames, StatsPlots, Revise, ACTRModels
# load stimuli
include("Stimuli.jl")
# load model functions
include("Paired_Model.jl")
# set seed for RNG
Random.seed!(589)
#######################################################################################
#                                   Generate Data
#######################################################################################
# true decay parameter
d = 0.5
# number of block repetitions
n_blocks = 8
# number of trials within a block
n_trials = 20
# fixed parameters
fixed_parms = (
    τ = -2.5,        # retrieval threshold
    noise = true,    # noise "on"
    bll = true,      # base-level learning "on"
    s = 0.2,         # scale parameter for activation noise
    lf = 0.4,        # latency factor
    ter = 0.535      # encoding, motor and conflict resolution time
)
# generate data
data = simulate(all_stimuli, fixed_parms, n_blocks, n_trials; d)
data[1]
#######################################################################################
#                                    Define Model
#######################################################################################
@model model(data, parms) = begin
    d ~ Beta(5, 5)
    data ~ Paired(d, parms)
end
#######################################################################################
#                                 Estimate Parameters
#######################################################################################
# number of samples
n_samples = 1000
delta = 0.85
# number of adaption samples 
n_adapt = 1000
# number of independent chains
n_chains = 4
# sampler object
specs = NUTS(n_adapt, delta)
# Start sampling.
chain = sample(
    model(data, fixed_parms),
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
ch = group(chain, :d)
font_size = 9
p1 = plot(ch, xaxis = font(font_size), yaxis = font(font_size), seriestype = (:traceplot),
    grid = false, size = (250, 100), titlefont = font(font_size))
p2 =
    plot(ch, xaxis = font(font_size), yaxis = font(font_size), seriestype = (:autocorplot),
        grid = false, size = (250, 100), titlefont = font(font_size))
p3 =
    plot(ch, xaxis = font(font_size), yaxis = font(font_size), seriestype = (:mixeddensity),
        grid = false, size = (300, 200), titlefont = font(font_size))
pcd = plot(p1, p2, p3, layout = (3, 1), size = (600, 600))
#######################################################################################
#                                  Posterior Predictive
#######################################################################################
preds = posterior_predictive(x -> show_learning(all_stimuli, fixed_parms; x...), chain, 100)
preds = vcat(preds..., source = :rep)
groups = groupby(preds, [:rep, :retrieved, :block])
mean_rts = combine(groups, :rt => mean => :mean_rt)
sort!(mean_rts, [:rep, :block, :retrieved])
fmean_rts = filter(x -> x.retrieved != :truncated, mean_rts)

p4 = @df fmean_rts plot(:block, :mean_rt, group = (:rep, :retrieved), xlabel = "Block",
    ylabel = "Mean RT", xaxis = font(9),
    yaxis = font(10), color = [:darkred :black], grid = false, size = (300, 200),
    titlefont = font(10), linewidth = 0.5,
    labeltitle = "Retrieved", legendtitlefontsize = 8, ylims = (0, 5.1), leg = false,
    alpha = 0.5)

groups = groupby(preds, :block)
accuracy = combine(groups, :retrieved => (x -> mean(x .== :retrieved)) => :accuracy)
sort!(accuracy)
p5 = @df accuracy plot(:block, :accuracy, xlabel = "Block", ylabel = "Mean Accuracy",
    xaxis = font(10),
    yaxis = font(10), color = :black, grid = false, size = (300, 200),
    titlefont = font(10), linewidth = 2,
    leg = false, legendtitlefontsize = 8)
savefig(p3, "Paired_Posterior.eps")
savefig(p4, "Paired_RTs.eps")
savefig(p5, "Paired_Accuracy.eps")
