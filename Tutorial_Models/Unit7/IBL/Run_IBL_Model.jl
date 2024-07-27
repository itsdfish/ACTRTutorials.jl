#######################################################################################
#                                   Load Packages
#######################################################################################
cd(@__DIR__)
using Pkg
Pkg.activate("../../..")
using Turing, StatsPlots, DataFrames, Revise, ACTRModels, Random
include("IBL_Model.jl")
Random.seed!(6025)
#######################################################################################
#                                   Generate Data
#######################################################################################
# get all three gambles
gamble_set = Gambles()
# number of trials
n_trials = 50
# memory decay parameter
d = 0.5
# decision noise parameter
ϕ = 0.2
# fixed parameters
parms = (τ = -10, s = 0.2, bll = true)
# generate data for all three gambles
data = map(x -> simulate(parms, x, n_trials; d, ϕ), gamble_set)
#######################################################################################
#                                    Define Model
#######################################################################################
@model model(data, parms, gambles) = begin
    d ~ Beta(10, 10)
    ϕ ~ truncated(Normal(0.2, 0.2), 0, Inf)
    data ~ IBL(d, ϕ, parms, gambles)
end
#######################################################################################
#                                 Estimate Parameters
#######################################################################################
# total samples
n_samples = 1000
# adaption samples
n_adapt = 1000
# number of chains
n_chains = 4
# sampler object
specs = NUTS(n_adapt, 0.65)
# start sampling.
chain = sample(
    model(data, parms, gamble_set),
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
font_size = 12
p1 = plot(ch, xaxis = font(font_size), yaxis = font(font_size), seriestype = (:traceplot),
    grid = false, size = (250, 100), titlefont = font(font_size))
p2 = plot(ch, xaxis = font(5), yaxis = font(font_size), seriestype = (:autocorplot),
    grid = false, size = (250, 100), titlefont = font(font_size))
p3 =
    plot(ch, xaxis = font(font_size), yaxis = font(font_size), seriestype = (:mixeddensity),
        grid = false, size = (250, 100), titlefont = font(font_size))
pcd = plot(p1, p2, p3, layout = (3, 1), size = (600, 600))

ch = group(chain, :ϕ)
p1 = plot(ch, xaxis = font(font_size), yaxis = font(font_size), seriestype = (:traceplot),
    grid = false, size = (250, 100), titlefont = font(font_size))
p2 =
    plot(ch, xaxis = font(font_size), yaxis = font(font_size), seriestype = (:autocorplot),
        grid = false, size = (250, 100), titlefont = font(font_size))
p3 =
    plot(ch, xaxis = font(font_size), yaxis = font(font_size), seriestype = (:mixeddensity),
        grid = false, size = (250, 100), titlefont = font(font_size))
pcϕ = plot(p1, p2, p3, layout = (3, 1), size = (600, 600))
#######################################################################################
#                                  Posterior Predictive
#######################################################################################
posterior_pred(x) = posterior_a_rate(parms, gamble_set, n_trials; x...)
preds = posterior_predictive(x -> posterior_pred(x), chain, 1000)
preds = permutedims(hcat(preds...))
df = DataFrame(preds, :auto)
colnames =
    Dict(Symbol(string("x", i)) => Symbol(string("Gamble ", i)) for i = 1:length(gamble_set))
rename!(df, colnames)
df = DataFrames.stack(df)
titles = reshape(unique(df.variable), 1, length(gamble_set))
p4 = @df df histogram(:value, group = :variable, layout = (3, 1), xlims = (0, 1),
    ylims = (0, 10), color = :grey, xlabel = "A-Rate",
    norm = true, leg = false, title = titles, titlefontsize = 12, grid = false,
    size = (550, 500))
obs_a_rates = a_rate.(data)
vline!(obs_a_rates', color = :darkred)

preds = posterior_predictive(
    x -> simulate(parms, gamble_set[1], n_trials; x...),
    chain,
    1000,
    recurrence_indices
)
v1 = mapreduce(x -> x[1], vcat, preds)
v2 = mapreduce(x -> x[2], vcat, preds)
hist1 = histogram2d(v1, v2, bins = n_trials, xlabel = "Trials", ylabel = "Trials",
    title = "Gamble 1 Recurrence",
    xaxis = font(12), yaxis = font(12))

preds = posterior_predictive(
    x -> simulate(parms, gamble_set[2], n_trials; x...),
    chain,
    1000,
    recurrence_indices
)
v1 = mapreduce(x -> x[1], vcat, preds)
v2 = mapreduce(x -> x[2], vcat, preds)
hist2 = histogram2d(v1, v2, bins = n_trials, xlabel = "Trials", ylabel = "Trials",
    title = "Gamble 2 Recurrence",
    xaxis = font(12), yaxis = font(12))

preds = posterior_predictive(
    x -> simulate(parms, gamble_set[3], n_trials; x...),
    chain,
    1000,
    recurrence_indices
)
v1 = mapreduce(x -> x[1], vcat, preds)
v2 = mapreduce(x -> x[2], vcat, preds)
hist3 = histogram2d(v1, v2, bins = n_trials, xlabel = "Trials", ylabel = "Trials",
    title = "Gamble 3 Recurrence",
    xaxis = font(12), yaxis = font(12))
plot(hist1, hist2, hist3, layout = (3, 1), size = (400, 800))

preds = posterior_predictive(
    x -> simulate(parms, gamble_set[1], n_trials; x...),
    chain,
    1000,
    recurrence_indices
)
v1 = mapreduce(x -> x[1], vcat, preds)
v2 = mapreduce(x -> x[2], vcat, preds)
hist1 = histogram2d(v1, v2, bins = n_trials, xlabel = "Trials", ylabel = "Trials",
    title = "Gamble 1 Recurrence",
    xaxis = font(12), yaxis = font(12))

preds = posterior_predictive(
    x -> simulate(parms, gamble_set[2], n_trials; x...),
    chain,
    1000,
    recurrence_indices
)
v1 = mapreduce(x -> x[1], vcat, preds)
v2 = mapreduce(x -> x[2], vcat, preds)
hist2 = histogram2d(v1, v2, bins = n_trials, xlabel = "Trials", ylabel = "Trials",
    title = "Gamble 2 Recurrence",
    xaxis = font(12), yaxis = font(12))

preds = posterior_predictive(
    x -> simulate(parms, gamble_set[3], n_trials; x...),
    chain,
    1000,
    recurrence_indices
)
v1 = mapreduce(x -> x[1], vcat, preds)
v2 = mapreduce(x -> x[2], vcat, preds)
hist3 = histogram2d(v1, v2, bins = n_trials, xlabel = "Trials", ylabel = "Trials",
    title = "Gamble 3 Recurrence",
    xaxis = font(12), yaxis = font(12))
plot(hist1, hist2, hist3, layout = (3, 1), size = (400, 800))
