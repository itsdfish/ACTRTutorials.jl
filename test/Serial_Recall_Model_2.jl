cd(@__DIR__)
using Pkg 
Pkg.activate("../")
using SafeTestsets

@safetestset "Serial Recall Model 2" begin
    using ACTRModels, Test, Distributions, Random
    include("../Tutorial_Models/Unit3/Serial_Recall_2/Serial_Recall_Model_2.jl")
    Random.seed!(2525)
    δ = 1.0
    τ = -.5
    d = .5
    n_blocks = 500
    n_items = 10
    n_study = 2
    parms = (s = .3,mmp = true,noise = true,mmpFun = penalty, bll=true)
    data = map(x -> simulate(parms, n_study, n_items; δ, τ, d), 1:n_blocks);

    x = range(δ*.8, δ*1.2, length=100)
    y = map(x -> computeLL(parms, data, n_items; δ=x, τ, d), x)
    mxv,mxi = findmax(y)
    δ′ = x[mxi]
    @test δ′ ≈ δ atol = 5e-2

    x = range(τ*.8, τ*1.2, length=100)
    y = map(x -> computeLL(parms, data, n_items; δ, τ=x, d), x)
    mxv,mxi = findmax(y)
    τ′ = x[mxi]
    @test τ′ ≈ τ atol = 5e-2

    x = range(d*.8, d*1.2, length=100)
    y = map(x -> computeLL(parms, data, n_items; δ, τ, d=x), x)
    mxv,mxi = findmax(y)
    d′ = x[mxi]
    @test d′ ≈ d atol = 2e-2
end
