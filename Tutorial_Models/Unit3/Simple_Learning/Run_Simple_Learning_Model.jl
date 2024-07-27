#######################################################################################
#                                   Load Packages
#######################################################################################
cd(@__DIR__)
using Pkg
Pkg.activate("../../..")
using Turing, StatsPlots, Revise, ACTRModels
include("LearningModel.jl")
Random.seed!(99051)
#######################################################################################
#                                   Generate Data
#######################################################################################
n_trials = 50
d = 0.5
parms = (τ = 0.5, s = 0.4, bll = true, noise = true)
temp = simulate(parms, n_trials; d)
data = vcat(temp...)
#######################################################################################
#                                    Define Model
#######################################################################################
@model model(data, parms) = begin
    d ~ Beta(5, 5)
    data ~ Retrieval(d, parms)
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
#                                      Summarize
#######################################################################################
println(chain)
#######################################################################################
#                                         Plot
#######################################################################################
pyplot()
ch = group(chain, :d)
font_size = 12
p1 = plot(ch, xaxis = font(font_size), yaxis = font(font_size), seriestype = (:traceplot),
    grid = false, size = (250, 100), titlefont = font(font_size))
p2 =
    plot(ch, xaxis = font(font_size), yaxis = font(font_size), seriestype = (:autocorplot),
        grid = false, size = (250, 100), titlefont = font(font_size))
p3 =
    plot(ch, xaxis = font(font_size), yaxis = font(font_size), seriestype = (:mixeddensity),
        grid = false, size = (250, 100), titlefont = font(font_size))
pcτ = plot(p1, p2, p3, layout = (3, 1), size = (600, 600))
#######################################################################################
#                                  Posterior Predictive
#######################################################################################
preds =
    posterior_predictive(x -> simulate(parms, n_trials; x...), chain, 100, learning_block)
p4 = plot(1:5, preds, xlabel = "Block", ylabel = "Accuracy", leg = false, grid = false,
    xaxis = font(font_size),
    yaxis = font(font_size), size = (600, 300), titlefont = font(font_size),
    color = :grey, linewidth = 1)
mean_pred = mean(hcat(preds...), dims = 2)
plot!(p4, 1:5, mean_pred, color = :black, linewidth = 1.5)
