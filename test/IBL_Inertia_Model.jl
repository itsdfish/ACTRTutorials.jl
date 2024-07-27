cd(@__DIR__)
using Pkg
Pkg.activate("../")
using SafeTestsets

@safetestset "IBL Inertia Model" begin
    using ACTRModels, Test, Distributions
    include("../Tutorial_Models/Unit7/IBL_Inertia/IBL_Inertia_Model.jl")
    Random.seed!(5045)
    d = 0.5
    ϕ = 0.2
    ρ = 0.25
    gambles = [Gambles() for i = 1:5]
    gambles = vcat(gambles...)
    n_trials = 200
    parms = (τ = -10, s = 0.2, bll = true)
    data = map(x -> simulate(parms, x, n_trials; d, ϕ, ρ), gambles)
    x = 0.3:0.01:0.7
    y = map(x -> computeLL(parms, gambles, data; d = x, ϕ, ρ), x)
    mv, mi = findmax(y)
    d′ = x[mi]
    @test d′ ≈ d atol = 0.05
    x = 0.1:0.01:0.4
    y = map(x -> computeLL(parms, gambles, data; d, ϕ = x, ρ), x)
    mv, mi = findmax(y)
    ϕ′ = x[mi]
    @test ϕ′ ≈ ϕ atol = 0.05
    x = 0.1:0.01:0.4
    y = map(x -> computeLL(parms, gambles, data; d, ϕ, ρ = x), x)
    mv, mi = findmax(y)
    ρ′ = x[mi]
    @test ρ′ ≈ ρ atol = 0.05
end
