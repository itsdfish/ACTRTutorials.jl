cd(@__DIR__)
using Pkg 
Pkg.activate("../")
using SafeTestsets

@safetestset "Guessing Mixture" begin
    using ACTRModels, Test, Parameters, Random
    include("../Tutorial_Models/Unit3/Guessing_Mixture/Guessing_Mixture.jl")
    Random.seed!(5045)
    n_trials = (t = 80000,f = 20000)
    τ = .5
    θg = .8
    fixed_parms = (blc = 1.5,s = .4)
    data = simulate(fixed_parms, n_trials; τ=τ, θg=θg)
    x = .01:.01:1
    y = map(x -> computeLL(fixed_parms, data; τ=x, θg=θg), x)
    mxv,mxi = findmax(y)
    τ′ = x[mxi]
    @test τ ≈ τ′ atol = 5e-2

    x = .01:.01:1
    y = map(x -> computeLL(fixed_parms, data;τ=τ,θg=x), x)
    mxv,mxi = findmax(y)
    θg′ = x[mxi]
    @test θg ≈ θg′ atol = 5e-2
end
