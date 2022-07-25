cd(@__DIR__)
using Pkg 
Pkg.activate("../")
using SafeTestsets

@safetestset "Simple PVT" begin
    using Distributions, KernelDensity, Test, Parameters, Random
    include("../Tutorial_Models/Unit5/Simple_PVT/model_functions.jl")
    Random.seed!(40132)
    υ = 4.0
    τ = 3.8
    λ = .98
    γ = .05
    
    n_trials = 100_000
    rts = simulate(υ, τ, λ, γ, n_trials)
    x = .08:.001:.5
    sim_dist = kde(rts)
    sim_dens = pdf(sim_dist, x)
    
    dens = map(x->pvt_log_like(x; υ, τ, λ, γ) |> exp, x)
    mu = mean(abs.(dens .- sim_dens))
    sd = mean(abs.(dens .- sim_dens))
    @test mu < .05 
    @test sd < .07
end
