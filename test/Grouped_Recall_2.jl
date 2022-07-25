cd(@__DIR__)
using Pkg 
Pkg.activate("../")
using SafeTestsets

@safetestset "Grouped Recall 2" begin
    using ACTRModels, Test, Distributions, Random, Combinatorics
    include("../Tutorial_Models/Unit8/Grouped_Recall_2/Grouped_Recall_2.jl")
    Random.seed!(54384)
    parms = (s = .15,τ = -.5,noise = true,mmp = true,mmp_fun = sim_fun)
    δ = 1.0
    Data = map(x -> simulate(;δ, parms...), 1:200)
    x = .2:.01:1.8
    y = map(x -> computeLL(Data, parms; δ=x), x)
    mxv,mxi = findmax(y)
    δ′ = x[mxi]
    @test δ′ ≈ δ atol = 1e-1
end
