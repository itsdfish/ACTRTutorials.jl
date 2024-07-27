cd(@__DIR__)
using Pkg
Pkg.activate("../")
using SafeTestsets

@safetestset "Likelihood" begin
    cd(@__DIR__)
    using Pkg
    Pkg.activate("../")
    using Test, ACTRModels, Distributions, FFTDists
    include("../Tutorial_Models/Unit6/Semantic_FFT/Semantic_FFT_Model.jl")
    include("../Tutorial_Models/Unit6/Semantic_FFT/model_functions.jl")
    Random.seed!(44154)
    #######################################################################################
    #                                   Generate Data
    #######################################################################################
    blc = 1.0
    δ = 1.0
    parms = (noise = true, τ = 0.0, s = 0.3, mmp = true)
    stimuli = get_stimuli()
    n_reps = 2_000
    data = map(s -> simulate(parms, s, n_reps; blc, δ), stimuli)

    x = range(0.8 * blc, 1.2 * blc, length = 100)
    y = map(x -> loglike(data, parms, x, δ), x)
    mxv, mxi = findmax(y)
    blc′ = x[mxi]
    @test blc′ ≈ blc rtol = 0.02

    x = range(0.8 * δ, 1.2 * δ, length = 100)
    y = map(x -> loglike(data, parms, blc, x), x)
    mxv, mxi = findmax(y)
    δ′ = x[mxi]
    @test δ′ ≈ δ rtol = 0.02
end

@safetestset "Semantic FFT Zero Chains" begin
    using Test, ACTRModels, Distributions, FFTDists, KernelDensity
    include("../Tutorial_Models/Unit6/Semantic_FFT/Semantic_FFT_Model.jl")
    include("../Tutorial_Models/Unit6/Semantic_FFT/model_functions.jl")
    Random.seed!(354301)
    #######################################################################################
    #                                   Generate Data
    #######################################################################################
    blc = 1.0
    δ = 1.0
    parms = (noise = true, τ = 0.0, s = 0.2, mmp = true)
    stimuli = get_stimuli()
    n_reps = 100_000
    stimulus = stimuli[1]
    data = simulate(parms, stimulus, n_reps; blc, δ)
    sim_dist_no = kde(data.no_rts)
    sim_dist_yes = kde(data.yes_rts)
    p_correct = length(data.yes_rts) / n_reps

    chunks = populate_memory()
    memory = Declarative(; memory = chunks)
    actr = ACTR(; declarative = memory, parms..., blc, δ, noise = false)
    x = 0.8:0.01:1.8

    exact_density_yes = map(i -> zero_chain_yes(actr, stimulus, i) |> exp, x)
    sim_density_yes = pdf(sim_dist_yes, x) * p_correct
    @test 0.15 > maximum(abs.(exact_density_yes .- sim_density_yes))

    exact_density_no = map(i -> zero_chain_no(actr, stimulus, i), x)
    exact_density_no .= exp.(exact_density_no)
    sim_density_no = pdf(sim_dist_no, x) * (1 - p_correct)
    @test 0.05 > maximum(abs.(exact_density_no .- sim_density_no))
end

@safetestset "Semantic FFT One Chain" begin
    using Test, ACTRModels, Distributions, FFTDists, KernelDensity
    include("../Tutorial_Models/Unit6/Semantic_FFT/Semantic_FFT_Model.jl")
    include("../Tutorial_Models/Unit6/Semantic_FFT/model_functions.jl")
    Random.seed!(354301)
    #######################################################################################
    #                                   Generate Data
    #######################################################################################
    blc = 1.0
    δ = 1.0
    parms = (noise = true, τ = 0.0, s = 0.2, mmp = true)
    stimuli = get_stimuli()
    n_reps = 100_000
    stimulus = stimuli[2]
    data = simulate(parms, stimulus, n_reps; blc = blc, δ = δ)
    sim_dist_no = kde(data.no_rts)
    sim_dist_yes = kde(data.yes_rts)
    p_correct = length(data.yes_rts) / n_reps

    chunks = populate_memory()
    memory = Declarative(; memory = chunks)
    actr = ACTR(; declarative = memory, parms..., blc, δ, noise = false)
    x = 0.8:0.01:2.5

    exact_density_yes = map(i -> one_chain_yes(actr, stimulus, i), x)
    exact_density_yes .= exp.(exact_density_yes)
    sim_density_yes = pdf(sim_dist_yes, x) * p_correct
    @test 0.15 > maximum(abs.(exact_density_yes .- sim_density_yes))

    exact_density_no = map(i -> one_chain_no(actr, stimulus, i), x)
    exact_density_no .= exp.(exact_density_no)
    sim_density_no = pdf(sim_dist_no, x) * (1 - p_correct)
    @test 0.05 > maximum(abs.(exact_density_no .- sim_density_no))
end

@safetestset "Semantic FFT all no one Chain" begin
    using Test, ACTRModels, Distributions, FFTDists, KernelDensity
    include("../Tutorial_Models/Unit6/Semantic_FFT/Semantic_FFT_Model.jl")
    include("../Tutorial_Models/Unit6/Semantic_FFT/model_functions.jl")
    Random.seed!(5445)
    #######################################################################################
    #                                   Generate Data
    #######################################################################################
    blc = 1.0
    δ = 1.0
    parms = (noise = true, τ = 0.0, s = 0.2, mmp = true)
    stimuli = get_stimuli()
    n_reps = 100_000
    stimulus = stimuli[3]
    data = simulate(parms, stimulus, n_reps; blc, δ)
    sim_dist_no = kde(data.no_rts)
    @test isempty(data.yes_rts)

    chunks = populate_memory()
    memory = Declarative(; memory = chunks)
    actr = ACTR(; declarative = memory, parms..., blc, δ, noise = false)
    x = 0.8:0.01:2.5

    exact_density_no = map(i -> one_chain_no(actr, stimulus, i), x)
    exact_density_no .= exp.(exact_density_no)
    sim_density_no = pdf(sim_dist_no, x)
    @test 0.10 > maximum(abs.(exact_density_no .- sim_density_no))
end

@safetestset "Semantic FFT all no two Chains" begin
    using Test, ACTRModels, Distributions, FFTDists, KernelDensity
    include("../Tutorial_Models/Unit6/Semantic_FFT/Semantic_FFT_Model.jl")
    include("../Tutorial_Models/Unit6/Semantic_FFT/model_functions.jl")
    Random.seed!(354301)
    #######################################################################################
    #                                   Generate Data
    #######################################################################################
    blc = 1.0
    δ = 1.0
    parms = (noise = true, τ = 0.0, s = 0.2, mmp = true)
    stimuli = get_stimuli()
    n_reps = 100_000
    stimulus = stimuli[4]
    data = simulate(parms, stimulus, n_reps; blc, δ)
    sim_dist_no = kde(data.no_rts)
    @test isempty(data.yes_rts)

    chunks = populate_memory()
    memory = Declarative(; memory = chunks)
    actr = ACTR(; declarative = memory, parms..., blc, δ, noise = false)
    x = 0.8:0.01:3.0

    exact_density_no = map(i -> two_chains_no(actr, stimulus, i), x)
    exact_density_no .= exp.(exact_density_no)
    sim_density_no = pdf(sim_dist_no, x)
    @test 0.10 > maximum(abs.(exact_density_no .- sim_density_no))
end
