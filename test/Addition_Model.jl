cd(@__DIR__)
using Pkg
Pkg.activate("../")
using SafeTestsets

@safetestset "Addition Model" begin
    using Test, Parameters, FFTDists, Random
    include("../Tutorial_Models/Unit4/Addition/Addition.jl")
    Random.seed!(6515)
    n_trials = 1000
    s = 0.3
    blc = 1.5
    data = simulate(n_trials; s, blc)
    filter!(x -> x < 2.9, data)
    x = 1:0.01:2
    y = map(x -> loglike(data, x, s), x)
    mxv, mxi = findmax(y)
    blc′ = x[mxi]
    @test blc′ ≈ blc atol = 5e-2
end
