cd(@__DIR__)
using Pkg
Pkg.activate("../")
using SafeTestsets

@safetestset "Simple Retrieval 3" begin
    using ACTRModels, Test, Parameters, Random
    include("../Tutorial_Models/Unit1/Simple_Retrieval_3/Simple_Retrieval_3.jl")
    include("../Utilities/Utilities.jl")
    Random.seed!(874)
    n_trials = 20_000
    n_items = 10
    stimuli = sample_stimuli(n_items, n_trials)
    τ = 0.5
    δ = 1.0
    parms = (blc = 1.5, s = 0.2, mmp = true)
    temp = simulate(n_items, stimuli, parms; δ, τ)
    data = unique_data(temp)

    x = range(0.8 * τ, 1.2 * τ, length = 100)
    y = map(x -> computeLL(parms, n_items, data; δ, τ = x), x)
    mxv, mxi = findmax(y)
    τ′ = x[mxi]
    @test τ ≈ τ′ atol = 5e-2

    x = range(0.8 * δ, 1.2 * δ, length = 100)
    y = map(x -> computeLL(parms, n_items, data; δ = x, τ), x)
    mxv, mxi = findmax(y)
    δ′ = x[mxi]
    @test δ ≈ δ′ atol = 2e-2

    c_data = filter(x -> x.matches, data)[1]
    p_correct = c_data.N / n_trials
    est_correct = computeLL(parms, n_items, [(matches = true, N = 1)]; δ, τ) |> exp
    @test p_correct ≈ est_correct atol = 5e-3
    # the probability incorrect for a specific non matching response
    p_incorrect = (1 - p_correct) / (n_items - 1)
    est_incorrect = computeLL(parms, n_items, [(matches = false, N = 1)]; δ, τ) |> exp
    @test p_incorrect ≈ est_incorrect atol = 5e-3
end
