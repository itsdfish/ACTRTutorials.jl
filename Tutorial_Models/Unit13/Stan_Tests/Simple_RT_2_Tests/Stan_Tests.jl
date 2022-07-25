cd(@__DIR__)
using Pkg
Pkg.activate("../../../../")
using SafeTestsets

@safetestset "Sreading Actitivation and Partial Matching" begin
    using ACTRModels, Test, Distributions, DataFrames, Random, CmdStan
    include("Stan_Test_Functions.jl")
    include("../../Simple_RT_2_Stan/Simple_RT_Model_2.jl")
    seed = 17644
    Random.seed!(seed)
    expose("Simple_RT_2.stan")
    # the number of trials
    n_trials = 50
    # true value of blc
    blc = 1.25
    # logistic scalar 
    s = 0.3
    # true value of τ
    τ = 0.5
    # perceptual-motor time
    ter = (0.05 + 0.085 + 0.05) + (0.05 + 0.06)
    σ = s * π / sqrt(3)
    parms = (noise = true,s = s,ter = ter)
    # generate data
    data = map(x -> simulate(parms; blc, τ), 1:n_trials)
    resp = map(x->x.resp, data)
    rts = map(x->x.rt, data)
    stanLLs = map(1:length(data)) do i
        stanLL(blc, τ, σ, ter, resp[i], rts[i])
    end
    turingLLs = map(x->computeLL(blc, τ, parms, [x]), data)
    @test turingLLs ≈ stanLLs atol=1e-6
end
