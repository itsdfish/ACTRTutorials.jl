cd(@__DIR__)
using Pkg
Pkg.activate("../")
using SafeTestsets

@safetestset "Simple RT Model 3" begin
    using ACTRModels, Test, Distributions, Random
    include("../Tutorial_Models/Unit2/Simple_RT_3/Simple_RT_Model_3.jl")
    Random.seed!(5045)
    n_trials = 10^4
    n_items = 10
    stimuli = sample_stimuli(n_items, n_trials)
    δ = 1.0
    τ = 0.5
    ter = (0.05 + 0.085 + 0.05) + (0.05 + 0.06)
    parms = (noise = true, ter = ter, blc = 1.25, s = 0.3, mmp = true)
    data = simulate(n_items, stimuli, parms, δ = δ, τ = τ)
    x = range(δ * 0.8, δ * 1.2, length = 100)
    y = map(x -> computeLL(n_items, parms, data, δ = x, τ = τ), x)
    mxv, mxi = findmax(y)
    δ′ = x[mxi]
    @test δ′ ≈ δ atol = 1e-1

    x = range(τ * 0.8, τ * 1.2, length = 100)
    y = map(x -> computeLL(n_items, parms, data, δ = δ, τ = x), x)
    mxv, mxi = findmax(y)
    τ′ = x[mxi]
    @test τ′ ≈ τ atol = 2e-1
end
