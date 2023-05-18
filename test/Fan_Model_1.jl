cd(@__DIR__)
using Pkg 
Pkg.activate("../")
using SafeTestsets

@safetestset "Fan Model 1" begin
    using ACTRModels, Test, Distributions, DataFrames, Random, StatsPlots
    include("../Tutorial_Models/Unit9/Fan_Model_1/Fan_Model_1.jl")
    include("../Tutorial_Models/Unit9/Common_Functions/Chunks.jl")
    include("../Tutorial_Models/Unit9/Common_Functions/Stimuli.jl")
    include("../Tutorial_Models/Unit9/Common_Functions/Utilities.jl")

    Random.seed!(5057)
    γ = 1.6
    Nblocks = 10^3
    parms = (blc=.3,τ=-.5,sa=true,noise=true,s=.3,ter=.845)
    temp = simulate(stimuli, slots, parms, Nblocks; γ=γ)
    data = vcat(temp...)
    x = range(γ*.5, γ*1.5, length=50)
    y = map(x -> computeLL(parms, slots, data; γ=x), x)
    mxv,mxi = findmax(y)
    γ′ = x[mxi]
    @test γ′≈ γ atol=1e-1
end