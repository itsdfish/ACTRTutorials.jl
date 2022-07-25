cd(@__DIR__)
using Pkg 
Pkg.activate("../")
using SafeTestsets

@safetestset "IBL Model" begin
    using ACTRModels, Test, Distributions, Random
    include("../Tutorial_Models/Unit7/IBL/IBL_Model.jl")
    Random.seed!(59952)
    d = 0.5
    ϕ = 0.2
    gambles = [Gambles() for i in 1:5]
    gambles = vcat(gambles...)
    n_trials = 100
    parms = (τ = -10,s = .2,bll = true)
    data = map(x -> simulate(parms, x, n_trials; d, ϕ), gambles)
    x = range(d*.8, d*1.2, length=50)
    y = map(x -> computeLL(parms, gambles, data; d=x, ϕ), x)
    mv,mi = findmax(y)
    d′ = x[mi]
    @test d′ ≈ d atol = .05
    x = range(ϕ*.8, ϕ*1.2, length=50)
    y = map(x -> computeLL(parms, gambles, data; d, ϕ=x), x)
    mv,mi = findmax(y)
    ϕ′ = x[mi]
    @test ϕ′ ≈ ϕ atol = .05
end
