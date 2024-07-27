cd(@__DIR__)
using Pkg
Pkg.activate("../")
using SafeTestsets

@safetestset "Grouped Recall 1" begin
    using ACTRModels, Test, Distributions, Random
    include("../Tutorial_Models/Unit8/Grouped_Recall_1/Grouped_Recall_1.jl")
    Random.seed!(5145)
    parms = (s = 0.15, τ = -0.5, noise = true, mmp = true, mmp_fun = sim_fun)
    δ = 1.0
    Data = map(x -> simulate(; δ, parms...), 1:500)
    x = 0.2:0.01:1.8
    y = map(x -> computeLL(Data, parms; δ = x), x)
    mxv, mxi = findmax(y)
    δ′ = x[mxi]
    @test δ′ ≈ δ atol = 1e-1
end
