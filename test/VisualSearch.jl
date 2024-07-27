cd(@__DIR__)
using Pkg
Pkg.activate("../")
using SafeTestsets

@safetestset "Visual Search" begin
    using VisualSearchACTR, Test, Parameters, Random
    include("../Tutorial_Models/Unit10/Visual_Search/Visual_Search_Model.jl")
    Random.seed!(95594)
    n_trials = 200
    topdown_weight = 0.66
    experiment = Experiment(; n_trials)
    fixed_parms = (noise = false, rnd_time = false)
    stimuli, all_fixations = simulate(experiment; topdown_weight, fixed_parms...)
    x = range(0.8 * topdown_weight, 1.2 * topdown_weight, length = 100)
    y = map(x -> computeLL(stimuli, all_fixations; topdown_weight = x, fixed_parms...), x)
    mxv, mxi = findmax(y)
    topdown_weight′ = x[mxi]
    @test topdown_weight ≈ topdown_weight′ atol = 4e-2
end
