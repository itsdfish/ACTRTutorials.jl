cd(@__DIR__)
using Pkg
Pkg.activate("../")
using SafeTestsets

@safetestset "Siegler Model" begin
    using ACTRModels, Test, Distributions, Random
    include("../Tutorial_Models/Unit3/Siegler_Choice/Siegler_Model_Choice.jl")
    include("../Utilities/Utilities.jl")
    Random.seed!(1505112)
    δ = 16.0
    τ = -0.45
    s = 0.5
    parms = (mmp = true, noise = true, mmp_fun = sim_fun, ter = 2.05)
    stimuli = [(num1 = 1, num2 = 1), (num1 = 1, num2 = 2), (num1 = 1, num2 = 3),
        (num1 = 2, num2 = 2),
        (num1 = 2, num2 = 3), (num1 = 3, num2 = 3)]
    temp = mapreduce(x -> simulate(stimuli, parms; δ = δ, τ = τ, s = s), vcat, 1:500)
    data = unique_data(temp)
    data = vcat(data...)

    x = range(0.8 * δ, 1.2 * δ, length = 100)
    y = map(x -> computeLL(parms, data; δ = x, τ = τ, s = s), x)
    mxv, mxi = findmax(y)
    δ′ = x[mxi]
    @test δ′ ≈ δ atol = 1

    x = range(0.8 * τ, 1.2 * τ, length = 100)
    y = map(x -> computeLL(parms, data; δ = δ, τ = x, s = s), x)
    mxv, mxi = findmax(y)
    τ′ = x[mxi]
    @test τ′ ≈ τ atol = 0.1

    x = range(0.8 * s, 1.2 * s, length = 100)
    y = map(x -> computeLL(parms, data; δ = δ, τ = τ, s = x), x)
    mxv, mxi = findmax(y)
    s′ = x[mxi]
    @test s′ ≈ s atol = 0.05

    n_trials = 1e4
    stimuli = [(num1 = 2, num2 = 2)]
    temp = mapreduce(x -> simulate(stimuli, parms; δ = δ, τ = τ, s = s), vcat, 1:n_trials)
    data = unique_data(temp)
    data = vcat(data...)
    likelihood(data) = exp(computeLL(parms, [data]; δ = δ, τ = τ, s = s) / data.N)
    simVals = map(x -> x.N / n_trials, data)
    actVals = map(x -> likelihood(x), data)
    @test simVals ≈ actVals atol = 2e-2
end
