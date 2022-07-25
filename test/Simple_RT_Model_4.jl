cd(@__DIR__)
using Pkg 
Pkg.activate("../")
using SafeTestsets

@safetestset "Simple RT Model 4" begin
    using ACTRModels, Test, Distributions, Random
    include("../Tutorial_Models/Unit2/Simple_RT_4/Simple_RT_Model_4.jl")
    Random.seed!(5045)
    n_trials = 10^4
    n_items = 10
    stimuli = sample_stimuli(n_items, n_trials)
    δ = 1.0
    s = .3
    τ = -.5
    ter = (.05 + .085 + .05) + (.05 + .06)
    parms = (noise = true,ter = ter, blc=1.25, mmp=true)
    data = simulate(n_items, stimuli, parms; δ, s, τ)
    x = range(δ*.8, δ*1.2, length=100)
    y = map(x -> computeLL(n_items, parms, data; δ=x, s, τ), x)
    mxv,mxi = findmax(y)
    δ′ = x[mxi]
    @test δ′ ≈ δ atol = 1e-2

    x = range(s*.8, s*1.2, length=100)
    y = map(x -> computeLL(n_items, parms, data; δ, s=x, τ), x)
    mxv,mxi = findmax(y)
    s′ = x[mxi]
    @test s′ ≈ s atol = 2e-1

    x = range(τ*.8, τ*1.2, length=100)
    y = map(x -> computeLL(n_items, parms, data; δ, s, τ=x), x)
    mxv,mxi = findmax(y)
    τ′ = x[mxi]
    @test τ′ ≈ τ atol = 2e-1
end
