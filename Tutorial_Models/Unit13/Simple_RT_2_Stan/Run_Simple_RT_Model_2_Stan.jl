#######################################################################################
#                                   Load Packages
#######################################################################################
cd(@__DIR__)
using Pkg
Pkg.activate("../../../")
using CmdStan, Distributions, Random, StatsPlots, ACTRModels, CSV, DataFrames, MCMCChains
# add your path here if not set up on machine's path
proj_dir = pwd()
n_chains = 4
include("Simple_RT_Model_2.jl")
seed = 54841
Random.seed!(seed)
#######################################################################################
#                                   Generate Data
#######################################################################################
# the number of trials
n_trials = 50
# true value of blc
blc = 1.25
# logistic scalar 
s = 0.3
# true value of τ
τ = 0.5
# perceptual-motor time
ter = (0.05 + 0.085 + 0.05) + (0.05 + 0.06)
parms = (noise = true, s = s, ter = ter)
# generate data
data = map(x -> simulate(parms; blc, τ), 1:n_trials)
# extract responses
resp = map(x -> x.resp, data)
# extract rts 
rts = map(x -> x.rt, data)
# add data to dictionary for Stan
stan_input =
    Dict("ter" => ter, "s" => s, "resp" => resp, "rts" => rts, "n_obs" => length(data))
#######################################################################################
#                                     Load Model
#######################################################################################
stream = open("Simple_RT_Model_2.stan", "r")
Model = read(stream, String)
close(stream)
stanmodel = Stanmodel(Sample(save_warmup = false, num_warmup = 1000,
        num_samples = 1000, thin = 1), nchains = n_chains, name = "Simple_RT_2", model = Model,
    printsummary = false, output_format = :mcmcchains, random = CmdStan.Random(seed))
#######################################################################################
#                                     Run Model
#######################################################################################
rc, chain, cnames = stan(stanmodel, stan_input, proj_dir)
chain = replacenames(chain, Dict(:tau => :τ))
#######################################################################################
#                                     Plot
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
