cd(@__DIR__)
using Pkg
Pkg.activate("../")
using SafeTestsets

@safetestset "Simple Retrieval" begin
    using ACTRModels, Test, Parameters, Random
    include("../Tutorial_Models/Unit1/Simple_Retrieval_1/Simple_Retrieval_1.jl")
    Random.seed!(5045)
    n_trials = 10^5
    τ = 0.5
    parms = (blc = 1.5, s = 0.4)
    Nᵣ = simulate(parms, n_trials; τ)
    x = range(0.8 * τ, 1.2 * τ, length = 100)
    y = map(x -> computeLL(parms, n_trials, Nᵣ; τ = x), x)
    mxv, mxi = findmax(y)
    τ′ = x[mxi]
    @test τ ≈ τ′ atol = 2e-2
end
