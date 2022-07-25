cd(@__DIR__)
using Pkg 
Pkg.activate("../")
using SafeTestsets

@safetestset "Simple RT Model 2" begin
    using ACTRModels, Test, Distributions, Random
    include("../Tutorial_Models/Unit2/Simple_RT_2/Simple_RT_Model_2.jl")
    Random.seed!(5045)
    Ntrials = 10^4
    blc = 1.5
    τ = -.5
    ter = (.05 + .085) + .05 + (.06 + .05)
    parms = (noise = true,s = .3,ter = ter)
    data = map(x -> simulate(parms; blc=blc, τ=τ), 1:Ntrials)
    x = range(blc*.8, blc*1.2, length=100)
    y = map(x -> computeLL(x, τ, parms, data), x)
    mxv,mxi = findmax(y)
    blc′ = x[mxi]
    @test blc′ ≈ blc atol = 1e-1

    x = range(τ*.8, τ*1.2, length=100)
    y = map(x -> computeLL(blc, x, parms, data), x)
    mxv,mxi = findmax(y)
    τ′ = x[mxi]
    @test τ′ ≈ τ atol = 1e-1
end
