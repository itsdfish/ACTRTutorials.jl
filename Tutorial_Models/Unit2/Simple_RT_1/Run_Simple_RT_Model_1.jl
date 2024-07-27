#######################################################################################
#                                   Load Packages
#######################################################################################
cd(@__DIR__)
using Pkg
Pkg.activate("../../..")
using Turing, StatsPlots, Revise, ACTRModels, Distributions
include("Simple_RT_Model_1.jl")
Random.seed!(6650)
#######################################################################################
#                                   Generate Data
#######################################################################################
n_trials = 50
blc = 1.5
ter = (0.05 + 0.085) + 0.05 + (0.06 + 0.05)
parms = (noise = true, τ = -10.0, s = 0.3, ter = ter)
data = map(x -> simulate(parms; blc), 1:n_trials)
#######################################################################################
#                                    Define Model
#######################################################################################
@model model(data, parms) = begin
    blc ~ Normal(1.5, 0.5)
    data ~ RT(blc, parms)
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
    sample(model(data, s, ter), specs, MCMCThreads(), n_samples, n_chains, progress = true)
#######################################################################################
#                                      Summarize
#######################################################################################
println(chain)
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
#######################################################################################
#                                  Posterior Predictive
#######################################################################################
preds = posterior_predictive(x -> simulate(parms; x...), chain, 1000)
p4 = histogram(preds, xlabel = "Reaction Time (seconds)", ylabel = "Density",
    xaxis = font(font_size), yaxis = font(font_size),
    grid = false, norm = true, color = :grey, leg = false, size = (600, 300),
    titlefont = font(font_size), xlims = (0, 1.5))

# sample blc values from posterior distribution
blcs = sample(chain[:blc], 10)
# set time step for x-axis
times = 0.01:0.01:1.5
# create a kernel density distribution object
posterior_dist = kde(Array(chain))
# estimate the posterior densities for blcs empirically
posterior_densities = pdf(posterior_dist, blcs)
# weight likelihood by posterior density
f(blc, w) = map(x -> computeLL([x]; blc = blc, parms...) |> exp, times) * w
predictive_density = map((x, y) -> f(x, y), blcs, posterior_densities)
# plot mixture components
plot(times, predictive_density, grid = false, xlabel = "Reaction Time (seconds)",
    ylabel = "Density",
    legendtitle = "blc", label = round.(blcs', digits = 3), size(800, 300))
