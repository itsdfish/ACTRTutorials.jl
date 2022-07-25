#######################################################################################
#                                   Load Packages
#######################################################################################
cd(@__DIR__)
using Pkg
Pkg.activate("../..")
using Turing, Distributions, StatsPlots, ACTRModels
#######################################################################################
#                                   Generate Data
#######################################################################################
n = 50
θ = 0.5
k = rand(Binomial(n, θ))
#######################################################################################
#                                    Define Model
#######################################################################################
@model model(n, k) = begin
    θ ~ Beta(5, 5)
    k ~ Binomial(n, θ)
end
#######################################################################################
#                                 Estimate Parameters
#######################################################################################
n_samples = 1000
n_adapt = 1000
n_chains = 4
δ = 0.85
specs = NUTS(n_adapt, δ)
chain = sample(model(n, k), specs, MCMCThreads(), n_samples, n_chains, progress=true)
#######################################################################################
#                                 Plot Results
#######################################################################################
simulate(;θ) = rand(Binomial(n, θ))
preds = posterior_predictive(x -> simulate(;x...), chain, 10000)
p = histogram(layout=(2,1), size=(600,600))
histogram!(p, preds, xlabel="h", leg=false, color=:grey, grid=false,
    yaxis=font(10), xaxis=font(10), subplot=1, ylim=(0,1200), xlim=(10,40), 
    bar_width=1, title="Posterior Predictive")

y = rand(Binomial(n, k / n), 10000)
histogram!(y, leg=false, color=:darkred, grid=false, yaxis=font(10), xlabel="h",
    xaxis=font(10), subplot=2, ylim=(0,1200), xlim=(10,40), bar_width=1, title="Predictive")