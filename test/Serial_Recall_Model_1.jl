cd(@__DIR__)
using Pkg
Pkg.activate("../")
using SafeTestsets

@safetestset "Serial Recall Model 1" begin
    using ACTRModels, Test, Distributions, Random
    include("../Tutorial_Models/Unit3/Serial_Recall_1/Serial_Recall_Model_1.jl")
    Random.seed!(5045)
    δ = 1.0
    n_blocks = 100
    n_items = 10
    parms = (s = 0.3, τ = -100.0, mmp = true, noise = true, mmpFun = penalty)
    data = map(x -> simulate(parms, n_items; δ = δ), 1:n_blocks)

    x = range(δ * 0.8, δ * 1.2, length = 100)
    y = map(x -> computeLL(parms, data, n_items; δ = x), x)
    mxv, mxi = findmax(y)
    δ′ = x[mxi]
    @test δ′ ≈ δ atol = 3e-1
end
