cd(@__DIR__)
using Pkg
Pkg.activate("../")
using SafeTestsets

@safetestset "Semantic Model" begin
    using ACTRModels, Test, Parameters, Random, HypothesisTests
    include("../Tutorial_Models/Unit5/Semantic/Semantic_Model.jl")
    Random.seed!(054548)
    n_reps = 10_000
    blc = 1.0
    parms = (noise = true, τ = 0.0, s = 0.2, mmp = true, δ = 1.0)
    stimuli = get_stimuli()
    data = map(x -> simulate(parms, x, n_reps; blc = blc), stimuli)
    x = range(blc * 0.5, blc * 1.5, length = 50)
    y = map(x -> computeLL(parms, data; blc = x), x)
    mxv, mxi = findmax(y)
    blc′ = x[mxi]
    @test blc ≈ blc′ atol = 1e-1

    # Test whether the generative model matches lisp version of model
    data = map(x -> simulate_exact(parms, x, n_reps; blc = blc), stimuli)
    t1 = BinomialTest(data[1].k, n_reps, 8727 / n_reps)
    p1 = pvalue(t1)
    t2 = BinomialTest(data[2].k, n_reps, 7617 / n_reps)
    p2 = pvalue(t2)
    @test p1 > 0.05
    @test p2 > 0.05
end
