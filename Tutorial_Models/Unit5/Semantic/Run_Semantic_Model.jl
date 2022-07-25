#######################################################################################
#                                   Load Packages
#######################################################################################
cd(@__DIR__)
using Pkg
Pkg.activate("../../..")
using StatsPlots, ACTRModels, Distributions, Turing
include("Semantic_Model.jl")
Random.seed!(354301)
#######################################################################################
#                                   Generate Data
#######################################################################################
blc = 1.0
parms = (noise = true, τ = 0.0, s = 0.2, mmp = true, δ = 1.0)
stimuli = get_stimuli()
n_reps = 10
data = map(x -> simulate(parms, x, n_reps; blc), stimuli)
#######################################################################################
#                                    Define Model
#######################################################################################
@model model(data, parms) = begin
    blc ~ Normal(1.0, 1.0)
    data ~ Semantic(blc, parms)
end
#######################################################################################
#                                 Estimate Parameters
#######################################################################################
# Settings of the NUTS sampler.
n_samples = 1500
n_adapt = 1500
specs = NUTS(n_adapt, 0.65)
n_chains = 4
chain = sample(model(data, parms), specs, MCMCThreads(), n_samples, n_chains, progress=true)
#######################################################################################
#                                         Plot
#######################################################################################
pyplot()
posteriors = plot(chain, seriestype=:density, grid=false, titlefont=font(10),
    size=(300,175), xaxis=font(8), yaxis=font(8))
#######################################################################################
#                                  Posterior Predictive
#######################################################################################
font_size = 12
hit_rates(s) = posterior_predictive(x -> hit_rate(parms, s, n_reps; x...), chain, 1000)
preds = map(s -> hit_rates(s), stimuli)
predictive_plot = histogram(preds, xlabel="% Correct" ,ylabel="Probability", xaxis=font(font_size), yaxis=font(font_size),
    grid=false, color=:grey, leg=false, titlefont=font(font_size), xlims=(0,1.1),
    layout=(2,1), ylims=(0,0.4), normalize=:probability, size=(600,600), title=["Is a canary a bird?" "Is a canary an animal?"])