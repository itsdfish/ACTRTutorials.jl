cd(@__DIR__)
using Pkg 
Pkg.activate("../")
using SafeTestsets

@safetestset "Fan Model 2" begin
    using ACTRModels, Test, Distributions
    using StatsPlots, DataFrames
    include("../Tutorial_Models/Unit9/Fan_Model_2/Fan_Model_2.jl")
    include("../Tutorial_Models/Unit9/Common_Functions/Chunks.jl")
    include("../Tutorial_Models/Unit9/Common_Functions/Stimuli.jl")
    include("../Tutorial_Models/Unit9/Common_Functions/Utilities.jl")

    Random.seed!(6112015)

    δ = .5
    γ = 1.65
    n_blocks = 500
    parms = (blc=0.0, s=.2, τ=-.5, ter=.845, mmp=true, sa=true, noise=true)
    temp = simulate(stimuli, slots, parms, n_blocks; δ=δ, γ=γ)
    data = vcat(temp...)
    x = range(δ*.8, δ*1.2, length=50)
    y = map(x -> computeLL(parms, slots, data; δ=x, γ=γ), x)
    mxv,mxi = findmax(y)
    δ′ = x[mxi]
    @test δ′≈ δ atol=1e-1

    x = range(γ*.8, γ*1.2, length=50)
    y = map(x -> computeLL(parms, slots, data; δ=δ, γ=x), x)
    mxv,mxi = findmax(y)
    γ′ = x[mxi]
    @test γ′≈ γ atol=1e-1
end
