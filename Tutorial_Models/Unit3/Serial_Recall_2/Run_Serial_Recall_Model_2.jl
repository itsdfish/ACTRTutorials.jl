#######################################################################################
#                                   Load Packages
#######################################################################################
cd(@__DIR__)
using Pkg
Pkg.activate("../../..")
using Turing, StatsPlots, ACTRModels #,Revise
include("Serial_Recall_Model_2.jl")
Random.seed!(2920)
#######################################################################################
#                                   Generate Data
#######################################################################################
δ = 1.0
τ = -1.0
d = 0.5
n_blocks = 10
n_items = 10
n_study = 1
parms = (s = 0.3, mmp = true, noise = true, mmpFun = penalty, bll = true)
data = map(x -> simulate(parms, n_study, n_items; δ, τ, d), 1:n_blocks);
#######################################################################################
#                                    Define Model
#######################################################################################
@model model(data, parms, n_items) = begin
    δ ~ truncated(Normal(1.0, 0.5), 0, Inf)
    τ ~ Normal(-1.0, 0.5)
    d ~ Beta(5, 5)
    data ~ SerialRecall(δ, τ, d, parms, n_items)
end
#######################################################################################
#                                 Estimate Parameters
#######################################################################################
# Settings of the NUTS sampler.
n_samples = 1000
n_adapt = 1000
n_chains = 4
specs = NUTS(n_adapt, 0.8)
# Start sampling.
chain = sample(
    model(data, parms, n_items),
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

ch = group(chain, :d)
p1 = plot(ch, xaxis = font(font_size), yaxis = font(font_size), seriestype = (:traceplot),
    grid = false, size = (250, 100), titlefont = font(font_size))
p2 =
    plot(ch, xaxis = font(font_size), yaxis = font(font_size), seriestype = (:autocorplot),
        grid = false, size = (250, 100), titlefont = font(font_size))
p3 =
    plot(ch, xaxis = font(font_size), yaxis = font(font_size), seriestype = (:mixeddensity),
        grid = false, size = (250, 100), titlefont = font(font_size))
pcd = plot(p1, p2, p3, layout = (3, 1), size = (800, 600))
#######################################################################################
#                                  Posterior Predictive
#######################################################################################
preds = posterior_predictive(x -> transpostions(parms, n_study, n_items; x...), chain, 1000)
preds = vcat(preds...)
p4 = histogram(preds, xlabel = "Position Displacement", ylabel = "Probability",
    xaxis = font(font_size),
    yaxis = font(font_size), grid = false, normalize = :probability, color = :grey,
    leg = false, size = (800, 400),
    titlefont = font(font_size), linewidth = 0.3)

preds =
    posterior_predictive(x -> serial_position(parms, n_study, n_items; x...), chain, 1000)
preds = hcat(preds...)
sp_effect = mean(preds, dims = 2)
p5 = plot(1:n_items, sp_effect, xlabel = "Position", ylims = (0, 1), ylabel = "Accuracy",
    xaxis = font(font_size),
    yaxis = font(font_size), grid = false, color = :grey, leg = false,
    size = (800, 400),
    titlefont = font(font_size), linewidth = 1.5)
