#######################################################################################
#                                   Load Packages
#######################################################################################
cd(@__DIR__)
using Pkg
Pkg.activate("../../..")
using StatsPlots, Revise, ACTRModels, Distributions, Turing, DataFrames
include("Siegler_Model_Choice.jl")
include("../../../Utilities/Utilities.jl")
Random.seed!(794145)
#######################################################################################
#                                   Generate Data
#######################################################################################
# mismatch penalty
δ = 16.0
# retrieval threshold
τ = -0.45
# logistic scalar 
s = 0.5
parms = (mmp = true, noise = true, mmp_fun = sim_fun, ter = 2.05)
stimuli =
    [(num1 = 1, num2 = 1), (num1 = 1, num2 = 2), (num1 = 1, num2 = 3), (num1 = 2, num2 = 2),
        (num1 = 2, num2 = 3), (num1 = 3, num2 = 3)]
temp = mapreduce(x -> simulate(stimuli, parms; δ, τ, s), vcat, 1:5)
# get unique data points with counts to improve efficiency
data = unique_data(temp)
data = vcat(data...)
#######################################################################################
#                                    Define Model
#######################################################################################
@model model(data, parms) = begin
    δ ~ Normal(16, 8)
    τ ~ Normal(-0.45, 1)
    s ~ truncated(Normal(0.5, 0.5), 0.0, Inf)
    data ~ Siegler(δ, τ, s, parms)
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
chain =
    sample(model(data, parms), specs, MCMCThreads(), n_samples, n_chains, progress = true)
#######################################################################################
#                                         Plot
#######################################################################################
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
pcτ = plot(p1, p2, p3, layout = (3, 1), size = (800, 600))

ch = group(chain, :τ)
p1 = plot(ch, xaxis = font(font_size), yaxis = font(font_size), seriestype = (:traceplot),
    grid = false, size = (250, 100), titlefont = font(font_size))
p2 =
    plot(ch, xaxis = font(font_size), yaxis = font(font_size), seriestype = (:autocorplot),
        grid = false, size = (250, 100), titlefont = font(font_size))
p3 =
    plot(ch, xaxis = font(font_size), yaxis = font(font_size), seriestype = (:mixeddensity),
        grid = false, size = (250, 100), titlefont = font(font_size))
pcτ = plot(p1, p2, p3, layout = (3, 1), size = (800, 600))

ch = group(chain, :s)
p1 = plot(ch, xaxis = font(font_size), yaxis = font(font_size), seriestype = (:traceplot),
    grid = false, size = (250, 100), titlefont = font(font_size))
p2 =
    plot(ch, xaxis = font(font_size), yaxis = font(font_size), seriestype = (:autocorplot),
        grid = false, size = (250, 100), titlefont = font(font_size))
p3 =
    plot(ch, xaxis = font(font_size), yaxis = font(font_size), seriestype = (:mixeddensity),
        grid = false, size = (250, 100), titlefont = font(font_size))
pcτ = plot(p1, p2, p3, layout = (3, 1), size = (800, 600))
#######################################################################################
#                                  Posterior Predictive
#######################################################################################
preds = posterior_predictive(x -> simulate(stimuli, parms; x...), chain, 1000)
preds = vcat(vcat(preds...)...)
df = DataFrame(preds)
p5 = response_histogram(df, stimuli)
