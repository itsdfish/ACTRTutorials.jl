cd(@__DIR__)
using Pkg 
Pkg.activate("../")
using SafeTestsets

@safetestset "Simple Retrieval PDA" begin
    using ACTRModels, Test, Parameters, Random
    include("../Tutorial_Models/Unit11/Simple_Retrieval_PDA/Simple_Retrieval_1_PDA.jl")
    Random.seed!(5045)
    n_trials = 10^5
    τ = 0.5
    parms = (blc = 1.5,s = .4)
    k = simulate(parms, n_trials; τ)
    x = range(.8*τ, 1.2*τ, length=100)
    y = map(x -> loglike(k, x; n_trials, parms, n_sim = 10^4), x)
    mxv,mxi = findmax(y)
    τ′ = x[mxi]
    @test τ ≈ τ′ atol = 2e-2
end
