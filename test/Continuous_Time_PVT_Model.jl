cd(@__DIR__)
using Pkg 
Pkg.activate("../")
using SafeTestsets

@safetestset "Continous Time PVT" begin
    using ACTRModels, Test, Parameters, Random
    include("../Tutorial_Models/Unit5/Continuous_Time_PVT/Continuous_Time_PVT.jl")
    Random.seed!(7845)
    σ = 0.3
    υ = 1.5
    τ = 0.7
    parms = (σ = σ,υ = υ,τ = τ)
    n_trials = 10_000
    rts = simulate(parms, n_trials)
    
    x = range(σ*.8, σ*1.2, length=50)
    y = map(x-> computeLL(rts; υ, τ, σ=x), x)
    mxv,mxi = findmax(y)
    σ′ = x[mxi]
    @test σ ≈ σ′ atol=1e-2

    x = range(τ*.8, τ*1.2, length=50)
    y = map(x-> computeLL(rts; υ, τ=x, σ), x)
    mxv,mxi = findmax(y)
    τ′ = x[mxi]
    @test τ ≈ τ′ atol=2e-2

    x = range(υ * .8, υ * 1.2, length=50)
    y = map(x-> computeLL(rts; υ=x, τ, σ), x)
    mxv,mxi = findmax(y)
    υ′ = x[mxi]
    @test υ ≈ υ′ atol=2e-1
end
