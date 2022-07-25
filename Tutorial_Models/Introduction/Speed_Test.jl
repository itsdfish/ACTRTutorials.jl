cd(@__DIR__)
using Pkg
Pkg.activate("../..")
using Distributions, Plots, Random
Random.seed!(4)
include("CoinFlip_Functions.jl")

n_reps = 10^4
n = 10
θ = .5

data = rand(Binomial(10, .5), 100)

time_analytic = @elapsed map(x -> analytic(data, n, θ), 1:n_reps)
time_analytic /= n_reps
time_computational = fill(0.0, 2)
time_computational[1] = @elapsed map(x -> computational(data, θ, n, 1_000), 1:n_reps)
time_computational[1] /= n_reps
time_computational[2] = @elapsed map(x -> computational(data, θ, n, 10_000), 1:n_reps)
time_computational[2] /= n_reps


bar(["analytic"  "1k simulations"  "10k simulations"], [time_analytic  time_computational...], 
     fillcolor=[:darkred :grey :darkgreen], fillrange=1e-5, alpha=.7,grid=false, xaxis=font(14), yaxis=font(14), 
     leg=false, yscale=:log10, ylabel="Seconds (log10)", xrotation=45, size=(800,400))
