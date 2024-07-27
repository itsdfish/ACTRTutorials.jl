#######################################################################################
#                                   Load Packages
#######################################################################################
cd(@__DIR__)
using Pkg
Pkg.activate("../../..")
using StatsPlots, Random, ACTRModels, CSV, FFTDists, DifferentialEvolutionMCMC, MCMCChains
include("Count.jl")
Random.seed!(76121);
#######################################################################################
#                                   Generate Data
#######################################################################################
# number of simulated trials
n_trials = 50
# logistic scalar for activation noise
s = 0.3
# base level constant
blc = 1.5
# generate simulated data
data = simulate(n_trials; s = s, blc = blc);
#######################################################################################
#                                    Define Model
#######################################################################################
# prior distributions
priors = (
    blc = (Normal(1.5, 1),),
    s = (Truncated(Normal(0.3, 0.5), 0.0, Inf),)
)
# lower and upper bounds of each parameter
bounds = ((-Inf, Inf), (eps(), Inf))
# define model object
model = DEModel(; priors, model = loglike, data)
# define DE sampler object
de = DE(; priors, bounds, burnin = 1000)
#######################################################################################
#                                 Estimate Parameters
#######################################################################################
n_iter = 2000
chains = sample(model, de, MCMCThreads(), n_iter, progress = true)
println(chains)
#######################################################################################
#                                         Plot
#######################################################################################
pyplot()
ch = group(chain, :blc)
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

ch = group(chain, :s)
p1 = plot(ch, xaxis = font(font_size), yaxis = font(font_size), seriestype = (:traceplot),
    grid = false, size = (250, 100), titlefont = font(font_size))
p2 =
    plot(ch, xaxis = font(font_size), yaxis = font(font_size), seriestype = (:autocorplot),
        grid = false, size = (250, 100), titlefont = font(font_size))
p3 =
    plot(ch, xaxis = font(font_size), yaxis = font(font_size), seriestype = (:mixeddensity),
        grid = false, size = (250, 100), titlefont = font(font_size))
pcτ = plot(p1, p2, p3, layout = (3, 1), size = (600, 600))

post_plot = plot(chain, xaxis = font(font_size), yaxis = font(font_size),
    seriestype = (:pooleddensity),
    grid = false, titlefont = font(font_size), size = (600, 600), color = :gray,
    linewidth = 2)
#######################################################################################
#                                  Posterior Predictive
#######################################################################################
preds = posterior_predictive(x -> simulate(n_trials; x...), chain, 1000)
rts = vcat(preds...)
post_pred =
    histogram(rts, xlabel = "RT", ylabel = "Frequency", xaxis = font(7), yaxis = font(7),
        grid = false, color = :grey, leg = false, size = (600, 300), titlefont = font(7),
        xlims = (0.5, 2.5))

preds = posterior_predictive(x -> simulate(n_trials; x...), chain, 1000, mean)
rts = vcat(preds...)
post_pred =
    histogram(rts, xlabel = "RT", ylabel = "Frequency", xaxis = font(7), yaxis = font(7),
        grid = false, color = :grey, leg = false, size = (600, 300), titlefont = font(7),
        xlims = (0.5, 2.5))
