#######################################################################################
#                                   Load Packages
#######################################################################################
# set the working directory to the directory in which this file is contained
cd(@__DIR__)
# load the package manager
using Pkg
# activate the project environment
Pkg.activate("../../..")
# load the required packages
using StatsPlots, Distributions, Turing, Random, ACTRModels
include("Simple_PVT_Model.jl")
include("model_functions.jl")
Random.seed!(35701)
#######################################################################################
#                                   Generate Data
#######################################################################################
# initial utility value
υ = 4.0
λ = 0.98
fixed_parms = (τ = 3.5, γ = 0.05)
n_trials = 100
rts = simulate(; υ, λ, fixed_parms..., n_trials)
#######################################################################################
#                                    Define Model
#######################################################################################
@model model(rts, fixed_parms) = begin
    υ ~ Normal(4.0, 1.0)
    λ ~ Beta(98, 2)
    rts ~ SimplePVT(υ, λ, fixed_parms)
end
#######################################################################################
#                                 Estimate Parameters
#######################################################################################
# Settings of the NUTS sampler.
n_samples = 1000
n_adapt = 1000
specs = NUTS(n_adapt, 0.65)
n_chains = 4
chain = sample(
    model(rts, fixed_parms),
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
ch = group(chain, :υ)
p1 = plot(ch, xaxis = font(font_size), yaxis = font(font_size), seriestype = (:traceplot),
    grid = false, size = (250, 100), titlefont = font(font_size))
p2 =
    plot(ch, xaxis = font(font_size), yaxis = font(font_size), seriestype = (:autocorplot),
        grid = false, size = (250, 100), titlefont = font(font_size))
p3 =
    plot(ch, xaxis = font(font_size), yaxis = font(font_size), seriestype = (:mixeddensity),
        grid = false, size = (250, 100), titlefont = font(font_size))
pcτ = plot(p1, p2, p3, layout = (3, 1), size = (800, 600))

ch = group(chain, :λ)
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
preds =
    posterior_predictive(x -> simulate(; x..., fixed_parms..., n_trials), chain, 1000, mean)
preds = vcat(preds...)
histogram(preds, norm = true, grid = false, leg = false, size = (800, 400), xlabel = "Mean",
    ylabel = "Density",
    xlims = (0.1, 0.5), color = :darkgrey, title = "Posterior Predictive of Mean",
    xaxis = font(font_size),
    yaxis = font(font_size))
vline!([mean(rts)], color = :darkred, linewidth = 2)

preds =
    posterior_predictive(x -> simulate(; x..., fixed_parms..., n_trials), chain, 1000, std)
preds = vcat(preds...)
histogram(preds, norm = true, grid = false, leg = false, size = (800, 400),
    xlabel = "Standard Deviation", ylabel = "Density",
    xlims = (0, 0.5), color = :darkgrey,
    title = "Posterior Predictive of Standard Deviation", xaxis = font(font_size),
    yaxis = font(font_size))
vline!([std(rts)], color = :darkred, linewidth = 2)

preds = posterior_predictive(
    x -> simulate(; x..., fixed_parms..., n_trials),
    chain,
    1000,
    skewness
)
preds = vcat(preds...)
histogram(preds, norm = true, grid = false, leg = false, size = (800, 400),
    xlabel = "Skewness", ylabel = "Density",
    xlims = (0, 4), color = :darkgrey, title = "Posterior Predictive of Skewness",
    xaxis = font(font_size),
    yaxis = font(font_size))
vline!([skewness(rts)], color = :darkred, linewidth = 2)

preds = posterior_predictive(
    x -> simulate(; x..., fixed_parms..., n_trials),
    chain,
    1000,
    x -> mean(x .> 0.5)
)
preds = vcat(preds...)
histogram(preds, norm = true, grid = false, leg = false, size = (800, 400),
    xlabel = "Proportion of Lapses", ylabel = "Density",
    xlims = (0, 0.1), color = :darkgrey, title = "Posterior Predictive of Lapses",
    xaxis = font(font_size),
    yaxis = font(font_size))
vline!([mean(rts .> 0.5)], color = :darkred, linewidth = 2)
