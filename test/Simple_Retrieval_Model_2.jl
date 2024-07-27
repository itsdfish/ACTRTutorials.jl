cd(@__DIR__)
using Pkg
Pkg.activate("../")
using SafeTestsets

@safetestset "Simple Retrieval 2" begin
    using ACTRModels, Test, Parameters, Random
    include("../Tutorial_Models/Unit1/Simple_Retrieval_2/Simple_Retrieval_2.jl")
    include("../Utilities/Utilities.jl")
    Random.seed!(23340)
    n_trials = 10_000
    n_items = 10
    stimuli = sample_stimuli(n_items, n_trials)
    τ = 0.5
    δ = 1.0
    parms = (blc = 1.5, s = 0.2, mmp = true)
    temp = simulate(n_items, stimuli, parms; δ = δ, τ = τ)
    data = unique_data(temp)

    x = range(0.8 * τ, 1.2 * τ, length = 100)
    y = map(x -> computeLL(parms, n_items, data; δ = δ, τ = x), x)
    mxv, mxi = findmax(y)
    τ′ = x[mxi]
    @test τ ≈ τ′ atol = 2e-2

    x = range(0.8 * δ, 1.2 * δ, length = 100)
    y = map(x -> computeLL(parms, n_items, data; δ = x, τ = τ), x)
    mxv, mxi = findmax(y)
    δ′ = x[mxi]
    @test δ ≈ δ′ atol = 2e-2

    p_correct = data[1].N / n_trials
    est_correct = computeLL(parms, n_items, [(resp = :correct, N = 1)]; δ = δ, τ = τ) |> exp
    @test p_correct ≈ est_correct atol = 2e-3

    p_failure = data[3].N / n_trials
    est_failure = computeLL(parms, n_items, [(resp = :failure, N = 1)]; δ = δ, τ = τ) |> exp
    @test p_failure ≈ est_failure atol = 2e-3
end
