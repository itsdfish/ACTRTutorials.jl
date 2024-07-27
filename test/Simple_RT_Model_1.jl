cd(@__DIR__)
using Pkg
Pkg.activate("../")
using SafeTestsets

@safetestset "Simple RT Model 1" begin
    using ACTRModels, Test, Parameters, Distributions, Random
    include("../Tutorial_Models/Unit2/Simple_RT_1/Simple_RT_Model_1.jl")
    Random.seed!(5045)
    n_trials = 10^5
    blc = 1.5
    ter = (0.05 + 0.085) + (0.06 + 0.05)
    parms = (noise = true, τ = -10.0, ter = ter, s = 0.3)
    data = map(x -> simulate(parms; blc = blc), 1:n_trials)
    x = range(blc * 0.8, blc * 1.2, length = 100)
    y = map(x -> computeLL(data; blc = x, parms...), x)
    mxv, mxi = findmax(y)
    blc′ = x[mxi]
    @test blc′ ≈ blc atol = 3e-2
end
