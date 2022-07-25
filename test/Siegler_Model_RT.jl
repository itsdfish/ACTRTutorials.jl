cd(@__DIR__)
using Pkg 
Pkg.activate("../")
using SafeTestsets

@safetestset "Siegler Model RT" begin
    using ACTRModels, Test, Distributions, Random, StatsPlots
    include("../Tutorial_Models/Unit6/Siegler_RT/Siegler_Model_RT.jl")
    Random.seed!(187302)
    δ = 16.0
    τ = -.45
    s = .5
    parms = (mmp = true,noise = true,mmp_fun = sim_fun,ter = 2.05)
    stimuli = [(num1 = 1,num2 = 1), (num1 = 1,num2 = 2), (num1 = 1,num2 = 3), (num1 = 2,num2 = 2),
        (num1 = 2,num2 = 3), (num1 = 3,num2 = 3)]
    temp = map(x -> simulate(stimuli, parms; δ, τ, s), 1:500)
    data = vcat(vcat(temp...)...)

    x = range(.8 * δ, 1.2 * δ, length=100)
    y = map(x -> computeLL(parms, data; δ=x, τ, s), x)
    mxv,mxi = findmax(y)
    δ′ = x[mxi]
    @test δ′ ≈ δ atol = 1

    x = range(.8 * τ, 1.2 * τ, length=100)
    y = map(x -> computeLL(parms, data; δ, τ=x, s), x)
    mxv,mxi = findmax(y)
    τ′ = x[mxi]
    @test τ′ ≈ τ atol = .1

    x = range(.8 * s, 1.2 * s, length=100)
    y = map(x -> computeLL(parms, data; δ, τ, s=x), x)
    mxv,mxi = findmax(y)
    s′ = x[mxi]
    @test s′ ≈ s atol = .03
end
