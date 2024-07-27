cd(@__DIR__)
using Pkg
Pkg.activate("../../../../")
using SafeTestsets

@safetestset "Siegler Choice" begin
    using ACTRModels, Test, Distributions, DataFrames, Random, CmdStan
    include("Stan_Test_Functions.jl")
    include("../../Siegler_Model_Stan/Siegler_Model_Choice.jl")
    include("../../../../Utilities/Utilities.jl")
    seed = 8775
    Random.seed!(seed)
    #######################################################################################
    #                                   Generate Data
    #######################################################################################
    # mismatch penalty
    δ = 1.60
    # retrieval threshold
    τ = -0.45
    # logistic scalar 
    s = 0.5
    parms = (mmp = true, noise = true)
    stimuli = [(num1 = 1, num2 = 1), (num1 = 1, num2 = 2), (num1 = 1, num2 = 3),
        (num1 = 2, num2 = 2),
        (num1 = 2, num2 = 3), (num1 = 3, num2 = 3)]
    n_blocks = 5
    temp = mapreduce(x -> simulate(stimuli, parms; δ, τ, s), vcat, 1:n_blocks)
    # get unique data points with counts to improve efficiency
    data = unique_data(temp)
    data = vcat(data...)

    n_obs = length(data)
    choice_reps = map(x -> x.N, data)
    choice = map(x -> x.resp, data)

    chunks = populate_memory()
    n_slots = 3
    n_chunks = length(chunks)
    memory_slots = repeat([1:3;]', outer = n_chunks)
    memory_values = [[values(c.slots)...] for c in chunks]
    memory_values = hcat(memory_values...)'
    stimuli_values = [[d.num1, d.num2] for d in data]
    stimuli_values = hcat(stimuli_values...)'
    stimuli_slots = repeat([1:2;]', outer = n_obs)

    expose("Siegler_Choice_Model.stan")

    stanLLs = map(1:length(data)) do i
        stanLL(0.0, 0.0, δ, 0.0, 0, 0, 0, s, τ, 0, 1, 0, n_chunks, n_slots,
            memory_slots, memory_values,
            stimuli_slots[i, :], stimuli_values[i, :], choice[i], choice_reps[i])
    end
    turingLLs = map(x -> computeLL(parms, [x]; δ, τ, s), data)
    @test turingLLs ≈ stanLLs atol = 1e-6
end
