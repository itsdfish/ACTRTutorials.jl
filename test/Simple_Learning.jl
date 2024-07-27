cd(@__DIR__)
using Pkg
Pkg.activate("../")
using SafeTestsets

@safetestset "Simple Learning Model" begin
    using ACTRModels, Test, Parameters, Random
    include("../Tutorial_Models/Unit3/Simple_Learning/LearningModel.jl")
    Random.seed!(5045)
    n_trials = 100
    d = 0.5
    parms = (τ = 0.5, s = 0.3, bll = true, noise = true)
    data = simulate(parms, n_trials; d)
    x = 0.01:0.01:0.99
    y = map(x -> computeLL(parms, data; d = x), x)
    mxv, mxi = findmax(y)
    d′ = x[mxi]
    @test d ≈ d′ atol = 5e-2
end
