#######################################################################################
#                                   Load Packages
#######################################################################################
cd(@__DIR__)
using Pkg
Pkg.activate("../../..")
using Turing, StatsPlots, Revise, ACTRModels
include("Simple_RT_Model_2.jl")
Random.seed!(65401)
#######################################################################################
#                                   Generate Data
#######################################################################################
# the number of trials
n_trials = 50
# true value of blc
blc = 1.25
# true value of τ
τ = 0.5
# perceptual-motor time
ter = (0.05 + 0.085 + 0.05) + (0.05 + 0.06)
parms = (noise = true, s = 0.3, ter = ter)
# generate data
data = map(x -> simulate(parms; blc, τ), 1:n_trials);
#######################################################################################
#                                    Define Model
#######################################################################################
@model model(data, parms) = begin
    blc ~ Normal(1.25, 0.5)
    τ ~ Normal(0.5, 0.5)
    data ~ RT(blc, τ, parms)
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
#######################################################################################
#                                  Posterior Predictive
#######################################################################################
preds = posterior_predictive(x -> simulate(parms; x...), chain, 1000)
correct = filter(x -> x.resp == 1, preds)
rts = map(x -> x.rt, correct)
p_correct = mean(x -> x.resp == 1, preds)
correct_dist = histogram(rts, xlabel = "RT", ylabel = "Density", xaxis = font(font_size),
    yaxis = font(font_size),
    grid = false, norm = true, color = :grey, leg = false, size = (600, 300),
    title = "Correct", titlefont = font(font_size),
    xlims = (0, 1.5))
correct_dist[1][1][:y] *= p_correct

incorrect = filter(x -> x.resp == 2, preds)
rts = map(x -> x.rt, incorrect)
incorrect_dist = histogram(rts, xlabel = "RT", ylabel = "Density", xaxis = font(font_size),
    yaxis = font(font_size),
    grid = false, norm = true, color = :grey, leg = false, size = (600, 300),
    title = "Incorrect", titlefont = font(font_size),
    xlims = (0, 1.5))
incorrect_dist[1][1][:y] *= (1 - p_correct)
plot(correct_dist, incorrect_dist, layout = (2, 1), ylims = (0, 3), size = (800, 500))
