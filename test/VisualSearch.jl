cd(@__DIR__)
using Pkg 
Pkg.activate("../")
using SafeTestsets

@safetestset "Visual Search" begin
    using VisualSearchACTR, Test, Parameters, Random
    include("../Tutorial_Models/Unit10/Visual_Search/Visual_Search_Model.jl")
    Random.seed!(93)
    n_trials = 200
    topdown_weight = .66
    stimuli,all_fixations = simulate(;n_trials, topdown_weight);
    x = range(.8 * topdown_weight, 1.2 * topdown_weight, length=100)
    @time y = map(x->computeLL(stimuli, all_fixations; topdown_weight=x), x)
    mxv,mxi = findmax(y)
    topdown_weight′ = x[mxi]
    @test topdown_weight ≈ topdown_weight′ atol = 2e-2
end